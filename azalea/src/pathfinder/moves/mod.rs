pub mod basic;
pub mod parkour;
pub mod uncommon;

use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use azalea_block::{
    BlockState,
    properties::{Half, Open},
};
use azalea_client::{
    PhysicsState, SprintDirection, StartSprintEvent, StartWalkEvent, WalkDirection,
    interact::StartUseItemEvent, inventory::SetSelectedHotbarSlotEvent,
    mining::StartMiningBlockEvent,
};
use azalea_core::position::{BlockPos, Vec3};
use azalea_inventory::Menu;
use azalea_protocol::packets::game::s_interact::InteractionHand;
use azalea_registry::builtin::BlockKind;
use azalea_world::World;
use bevy_ecs::{entity::Entity, message::MessageWriter, system::Commands, world::EntityWorldMut};
use parking_lot::RwLock;
use tracing::debug;

use super::{
    DoorHandling, astar,
    custom_state::CustomPathfinderStateRef,
    mining::MiningCache,
    positions::RelBlockPos,
    world::{CachedWorld, is_block_state_passable, is_door_block_kind},
};
use crate::{
    auto_tool::best_tool_in_hotbar_for_block,
    bot::{JumpEvent, LookAtEvent},
    pathfinder::player_pos_to_block_pos,
};

type Edge = astar::Edge<RelBlockPos, MoveData>;

pub type SuccessorsFn = fn(&mut MovesCtx, RelBlockPos);

/// Re-implement certain bugs and quirks that Baritone has, and disable
/// movements that Baritone doesn't have.
///
/// Meant to help with debugging when directly comparing against Baritone.
pub const BARITONE_COMPAT: bool = false;

pub fn default_move(ctx: &mut MovesCtx, node: RelBlockPos) {
    basic::basic_move(ctx, node);
    parkour::parkour_move(ctx, node);
    uncommon::uncommon_move(ctx, node);
}

#[derive(Clone)]
pub struct MoveData {
    /// Use the context to determine what events should be sent to complete this
    /// movement.
    pub execute: &'static (dyn Fn(ExecuteCtx) + Send + Sync),
    /// Whether we've reached the target.
    pub is_reached: &'static (dyn Fn(IsReachedCtx) -> bool + Send + Sync),
}
impl Debug for MoveData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveData")
            // .field("move_kind", &self.move_kind)
            .finish()
    }
}

pub struct ExecuteCtx<'s, 'w1, 'w2, 'w3, 'w4, 'w5, 'w6, 'w7, 'a, 'ep> {
    pub entity: Entity,
    /// The node that we're trying to reach.
    pub target: BlockPos,
    /// The last node that we reached.
    pub start: BlockPos,
    pub position: Vec3,
    pub physics: &'a azalea_entity::Physics,
    pub is_currently_mining: bool,
    pub can_mine: bool,
    pub world: Arc<RwLock<World>>,
    pub menu: Menu,

    pub commands: &'a mut Commands<'w1, 's>,
    pub look_at_events: &'a mut MessageWriter<'w2, LookAtEvent>,
    pub sprint_events: &'a mut MessageWriter<'w3, StartSprintEvent>,
    pub walk_events: &'a mut MessageWriter<'w4, StartWalkEvent>,
    pub jump_events: &'a mut MessageWriter<'w5, JumpEvent>,
    pub start_mining_events: &'a mut MessageWriter<'w6, StartMiningBlockEvent>,
    pub start_use_item_events:
        &'a mut MessageWriter<'w7, azalea_client::interact::StartUseItemEvent>,
    pub executing_path: Option<&'ep mut super::ExecutingPath>,
    pub door_handling: &'a DoorHandling,
}

impl ExecuteCtx<'_, '_, '_, '_, '_, '_, '_, '_, '_, '_> {
    pub fn on_tick_start(&mut self) {
        self.set_sneaking(false);
    }

    pub fn look_at(&mut self, position: Vec3) {
        self.look_at_events.write(LookAtEvent {
            entity: self.entity,
            position: Vec3 {
                x: position.x,
                // look forward
                y: self.position.up(1.53).y,
                z: position.z,
            },
        });
    }

    pub fn look_at_exact(&mut self, position: Vec3) {
        self.look_at_events.write(LookAtEvent {
            entity: self.entity,
            position,
        });
    }

    pub fn sprint(&mut self, direction: SprintDirection) {
        self.sprint_events.write(StartSprintEvent {
            entity: self.entity,
            direction,
        });
    }

    pub fn walk(&mut self, direction: WalkDirection) {
        self.walk_events.write(StartWalkEvent {
            entity: self.entity,
            direction,
        });
    }

    pub fn jump(&mut self) {
        self.jump_events.write(JumpEvent {
            entity: self.entity,
        });
    }

    fn set_sneaking(&mut self, sneaking: bool) {
        self.commands
            .entity(self.entity)
            .queue(move |mut entity: EntityWorldMut<'_>| {
                if let Some(mut physics_state) = entity.get_mut::<PhysicsState>() {
                    physics_state.trying_to_crouch = sneaking;
                }
            });
    }
    pub fn sneak(&mut self) {
        self.set_sneaking(true);
    }

    pub fn jump_if_in_water(&mut self) {
        if self.physics.is_in_water() {
            self.jump();
        }
    }

    /// Returns whether this block could be mined.
    pub fn should_mine(&mut self, block: BlockPos) -> bool {
        let block_state = self.world.read().get_block_state(block).unwrap_or_default();
        should_mine_block_state(block_state, self.door_handling)
    }

    /// Mine the block at the given position.
    ///
    /// Returns whether the block is being mined.
    pub fn mine(&mut self, block: BlockPos) -> bool {
        if !self.can_mine {
            return false;
        }

        let block_state = self.world.read().get_block_state(block).unwrap_or_default();
        if is_block_state_passable(block_state, self.door_handling) {
            // block is already passable, no need to mine it
            return false;
        }

        let best_tool_result = best_tool_in_hotbar_for_block(block_state, &self.menu);
        debug!("best tool for {block_state:?}: {best_tool_result:?}");

        self.commands.trigger(SetSelectedHotbarSlotEvent {
            entity: self.entity,
            slot: best_tool_result.index as u8,
        });

        self.is_currently_mining = true;

        self.walk(WalkDirection::None);
        self.look_at_exact(block.center());
        self.start_mining_events.write(StartMiningBlockEvent {
            entity: self.entity,
            position: block,
            force: true,
        });

        true
    }

    /// Interact with the block at the given position (e.g. open a door).
    ///
    /// Returns whether the block is being interacted with.
    pub fn interact(&mut self, block: BlockPos) -> bool {
        let Some(executing_path) = self.executing_path.as_mut() else {
            return false;
        };

        let block_state = self.world.read().get_block_state(block).unwrap_or_default();
        let registry_block = BlockKind::from(block_state);

        let target_block = if is_door_block_kind(registry_block) {
            if block_state.property::<Half>() == Some(Half::Upper) {
                block.down(1)
            } else {
                block
            }
        } else {
            block
        };

        if executing_path.interacted_blocks.contains(&target_block) {
            return false;
        }

        let target_state = self
            .world
            .read()
            .get_block_state(target_block)
            .unwrap_or_default();
        if !should_interact_with_door_block_state(target_state) {
            return false;
        }

        let is_door = is_door_block_kind(BlockKind::from(target_state));

        // Prevent spamming interactions on doors due to delayed block state updates
        if is_door
            && executing_path
                .last_door_interactions
                .get(&target_block)
                .is_some_and(|&last_tick| executing_path.ticks_executing - last_tick < 40)
        {
            return false;
        }

        self.start_use_item_events.write(StartUseItemEvent {
            entity: self.entity,
            hand: InteractionHand::MainHand,
            force_block: Some(target_block),
        });

        // Don't add doors to interacted_blocks, since they can close and need to be
        // opened again
        if !is_door {
            executing_path.interacted_blocks.push(target_block);
        } else {
            executing_path
                .last_door_interactions
                .insert(target_block, executing_path.ticks_executing);
        }

        true
    }

    /// Mine the given block, but make sure the player is standing at the start
    /// of the current node first.
    pub fn mine_while_at_start(&mut self, block: BlockPos) -> bool {
        let horizontal_distance_from_start = (self.start.center() - self.position)
            .horizontal_distance_squared()
            .sqrt();
        let at_start_position = player_pos_to_block_pos(self.position) == self.start
            && horizontal_distance_from_start < 0.25;

        if self.should_mine(block) {
            if at_start_position {
                self.look_at(block.center());
                self.mine(block);
            } else {
                self.look_at(self.start.center());
                self.walk(WalkDirection::Forward);
            }
            true
        } else {
            false
        }
    }

    /// Interact with the given block (e.g. open a door), but make sure the
    /// player is standing at the start of the current node first.
    pub fn interact_while_at_start(&mut self, block: BlockPos) -> bool {
        let Some(_) = self.executing_path.as_ref() else {
            return false;
        };

        let horizontal_distance_from_start = (self.start.center() - self.position)
            .horizontal_distance_squared()
            .sqrt();
        let at_start_position = player_pos_to_block_pos(self.position) == self.start
            && horizontal_distance_from_start < 0.25;

        let should_interact = should_interact_with_door_block_state(self.get_block_state(block));
        if should_interact {
            let is_door = is_door_block_kind(BlockKind::from(self.get_block_state(block)));
            let can_interact_now = if is_door {
                // for doors, allow interaction if close enough to the block
                let distance_to_block =
                    self.position.horizontal_distance_squared_to(block.center());
                distance_to_block < 4.0 // within 2 blocks
            } else {
                at_start_position
            };

            if can_interact_now {
                self.look_at(block.center());
                self.interact(block);
                true
            } else {
                // walk towards the target
                self.look_at(block.center());
                self.walk(WalkDirection::Forward);
                true
            }
        } else {
            false
        }
    }

    pub fn get_block_state(&self, block: BlockPos) -> BlockState {
        self.world.read().get_block_state(block).unwrap_or_default()
    }
}

pub fn should_mine_block_state(block_state: BlockState, door_handling: &DoorHandling) -> bool {
    let block_kind = BlockKind::from(block_state);
    if is_block_state_passable(block_state, door_handling) || block_kind == BlockKind::Water {
        // block is already passable, no need to mine it
        return false;
    }

    true
}

/// Check if this block is a closed door that needs to be opened.
pub fn should_interact_with_door_block_state(block_state: BlockState) -> bool {
    let registry_block = BlockKind::from(block_state);
    if !is_door_block_kind(registry_block) {
        return false;
    }

    // Can't open iron or waxed copper doors/trapdoors by hand
    if matches!(
        registry_block,
        BlockKind::IronDoor
            | BlockKind::WaxedCopperDoor
            | BlockKind::WaxedExposedCopperDoor
            | BlockKind::WaxedWeatheredCopperDoor
            | BlockKind::WaxedOxidizedCopperDoor
            | BlockKind::IronTrapdoor
            | BlockKind::WaxedCopperTrapdoor
            | BlockKind::WaxedExposedCopperTrapdoor
            | BlockKind::WaxedWeatheredCopperTrapdoor
            | BlockKind::WaxedOxidizedCopperTrapdoor
    ) {
        return false;
    }

    // Check if the door is closed
    block_state.property::<Open>() == Some(false)
}

pub struct IsReachedCtx<'a> {
    /// The node that we're trying to reach.
    pub target: BlockPos,
    /// The last node that we reached.
    pub start: BlockPos,
    pub position: Vec3,
    pub physics: &'a azalea_entity::Physics,
}

/// Returns whether the entity is at the node and should start going to the
/// next node.
#[must_use]
pub fn default_is_reached(
    IsReachedCtx {
        position,
        target,
        physics,
        ..
    }: IsReachedCtx,
) -> bool {
    let block_pos = player_pos_to_block_pos(position);
    if block_pos == target {
        return true;
    }
    // it's fine if we go over the target while swimming
    if physics.is_in_water() && block_pos.down(1) == target {
        return true;
    }

    false
}

pub struct MovesCtx<'a> {
    pub edges: &'a mut Vec<Edge>,
    pub world: &'a CachedWorld,
    pub mining_cache: &'a MiningCache,
    pub custom_state: &'a CustomPathfinderStateRef,
    pub door_handling: &'a DoorHandling,
}
