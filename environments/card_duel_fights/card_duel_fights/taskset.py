"""Card Duel fights as a verifiers v1 taskset with a custom Env.

Multi-turn: the game engine plays the user. Each model reply is one action
(<action>play CARD</action> or <action>end turn</action>); the engine answers
with the new state until the fight ends. The reward is 0.5 + 0.5 x HP share
for a win, or 0.4 x share of enemy HP removed for a loss.

Modelled on the textarena taskset in the verifiers repository. Not yet run
against a real verifiers install.
"""

from typing import Literal

import verifiers.v1 as vf

from card_duel.fight import FightEnv, fight_dataset, make_fight

SYSTEM_PROMPT = ("Play the Card Duel fight. Each reply must contain exactly one action inside "
                 "<action></action>.")


class CardDuelFightConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"
    num_tasks: int = 200
    level: Literal[0, 1, 2, 3] = 0  # each level: enemy +15% HP and +1 damage per hit


class CardDuelFightData(vf.TaskData):
    info: dict


class CardDuelFightTask(vf.Task[CardDuelFightData]):
    pass


class CardDuelFightTaskset(vf.Taskset[CardDuelFightTask, CardDuelFightConfig]):
    def load(self) -> list[CardDuelFightTask]:
        cfg = self.config
        return [
            CardDuelFightTask(
                CardDuelFightData(
                    name=spec.id,
                    prompt=FightEnv(spec).intro(),
                    system_prompt=SYSTEM_PROMPT,
                    info={"seed": spec.seed, "level": spec.level},
                ),
                cfg.task,
            )
            for spec in fight_dataset(cfg.split, cfg.num_tasks, level=cfg.level)
        ]


class CardDuelFightEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class CardDuelFightEnv(vf.Env[CardDuelFightEnvConfig]):
    async def run(self, task, agents) -> None:
        info = task.data.info
        fight = FightEnv(make_fight(int(info["seed"]), int(info.get("level", 0))))
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()  # the task prompt holds the opening state
            while not segment.terminated:
                observation, _, done, _ = fight.step(segment.last_reply or "")
                if done:
                    break
                segment = await interaction.turn(observation)
            trace = interaction.trace
            trace.record_reward("fight_reward", fight.reward(), 1.0)
            trace.info["card_duel"] = {
                "won": fight.won,
                "turns": fight.battle.turn,
                "hp_left": max(fight.battle.player.hp, 0),
                "illegal_actions": fight.illegal,
            }
