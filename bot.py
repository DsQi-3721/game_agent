import asyncio
from poke_env.player import Player, RandomPlayer

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best)
        return self.choose_random_move(battle)

async def main():
    smart = MaxDamagePlayer(battle_format="gen9randombattle")
    rand  = RandomPlayer(battle_format="gen9randombattle")
    await smart.battle_against(rand, n_battles=100)
    print(f"MaxDamage vs Random: {smart.n_won_battles}/100")

asyncio.run(main())