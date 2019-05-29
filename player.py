import numpy as np
class Player(object):

    check_call_percentage=0
    bet_raise_percentage=0
    fold_percentage=0

    def make_bets(self,card_rank,possible_actions,round_pot_size):
        if len(possible_actions) == 3:
            rank = "Pair"
            if rank in card_rank or card_rank == "High Card":
                check_call_percentage = 55
                fold_percentage = 10

            else:
                check_call_percentage=50
                fold_percentage=5

            get_action = np.random.randint(0,100,1)
            if get_action <= fold_percentage:
                return "Fold", 0
            elif fold_percentage < get_action <= check_call_percentage:
                return "Check/Call",round_pot_size
            else:
                toss_for_bet_size =  np.random.randint(0,100,1)
                if toss_for_bet_size <=50:
                    bet_amount = 50
                else:
                    bet_amount=100
                return "Bet",bet_amount
        elif len(possible_actions) == 2:
            rank = "Pair"
            if rank in card_rank or card_rank == "High Card":
                fold_percentage = 45

            else:
                fold_percentage = 0.25

            get_action = np.random.randint(0, 100, 1)
            if get_action <= fold_percentage:
                return "Fold",0
            else:
                if possible_actions[1] == "Bet":
                    toss_for_bet_size  = np.random.randint(0,100,1)
                    if toss_for_bet_size <=50:
                        return possible_actions[1],50
                    else:
                        return possible_actions[1],100
                else:
                    return possible_actions[1],round_pot_size




