from keras import models, Model
from treys import Deck, Evaluator, Card
import numpy as np

deck = Deck()
evaluator = Evaluator()
flop = deck.draw(3)
player1_hand = deck.draw(2)
player2_hand = deck.draw(3)
turn = deck.draw(1)
river = deck.draw(1)
player1_complete_hand = flop + [turn] + [river] + player1_hand
player2_complete_hand = flop + [turn] + [river] + player2_hand


def get_action_position2(y_stage,other_player_action):
    if other_player_action == "Bet/Raise":
        out_of_bounds_index = 2
    index = 0
    max_gain_loss = y_stage[0]
    for i in range(1, len(y_stage)):
        if i != out_of_bounds_index:
            if max_gain_loss < y_stage[i]:
                index = i
                max_gain_loss = y_stage[i]
    if index == 0:
        return "Fold"
    elif index == 1:
        return "Check/Call"
    else:
        return "Bet/Raise"


def get_action(y_stage):
    index = 0
    max_gain_loss = y_stage[0]
    for i in range(1 ,len(y_stage)):
        if max_gain_loss < y_stage[i]:
            index = i
            max_gain_loss = y_stage[i]
    if index == 0:
        return "Fold"
    elif index == 1:
        return "Check/Call"
    else:
        return "Bet/Raise"


def convert_pot_to_numpy(total_pot):
    pot_array = np.zeros((4, 13))
    number_of_chips = int(total_pot / 25)
    if number_of_chips > 13:
        pot_array[1] = 1
        left_over_chips = number_of_chips - 13
        for i in range(0, left_over_chips):
            pot_array[2][i] = 1
    else:
        for i in range(0, number_of_chips):
            pot_array[1][i] = 1
    return pot_array


def convert_to_numpy_array(Card_str, all_card_array_player1,all_card_array_player2):
    if Card_str[0] == "J":
        index_1 = 9
    elif Card_str[0] == "Q":
        index_1 = 10
    elif Card_str[0] == "K":
        index_1 = 11
    elif Card_str[0] == "A":
        index_1 = 12
    elif Card_str[0] == "T":
        index_1 = 8
    else:
        index_1 = int(Card_str[0]) - 2

    if Card_str[1] == "s":
        index_2 = 0
    elif Card_str[1] == "c":
        index_2 = 1
    elif Card_str[1] == "h":
        index_2 = 2
    else:
        index_2 = 3

    new_card_array = np.zeros((4, 13))
    new_card_array[index_2][index_1] = 1
    all_card_array_player1[index_2][index_1] = 1
    all_card_array_player2[index_2][index_1] = 1
    return new_card_array, all_card_array_player1,all_card_array_player2

def convert_to_numpy_array_playercards(Card_str, all_card_array):
    if Card_str[0] == "J":
        index_1 = 9
    elif Card_str[0] == "Q":
        index_1 = 10
    elif Card_str[0] == "K":
        index_1 = 11
    elif Card_str[0] == "A":
        index_1 = 12
    elif Card_str[0] == "T":
        index_1 = 8
    else:
        index_1 = int(Card_str[0]) - 2

    if Card_str[1] == "s":
        index_2 = 0
    elif Card_str[1] == "c":
        index_2 = 1
    elif Card_str[1] == "h":
        index_2 = 2
    else:
        index_2 = 3

    new_card_array = np.zeros((4, 13))
    new_card_array[index_2][index_1] = 1
    all_card_array[index_2][index_1] = 1

    return new_card_array, all_card_array


def betting(model_player1,model_player2,flop1_array,flop2_array,flop3_array,turn_array,river_array,Card1_array_player1,
            Card2_array_player1,Card1_array_player2,Card2_array_player2,all_card_array_player1,all_card_array_player2,
            stage_initial_potsize):
    pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
    state_array_player1 = np.stack((flop1_array,flop2_array,flop3_array,turn_array,river_array,Card1_array_player1,
                                    Card2_array_player1,all_card_array_player1,pot_array_stage))
    y_player1 = Model.predict(model_player1,x=state_array_player1)
    action_player1 = get_action(y_player1)
    if action_player1 == "Fold":
        return "Player 2",stage_initial_potsize,0,0
    elif action_player1 == "Check/Call":
        state_array_player2 = np.stack((flop1_array,flop2_array,flop3_array,turn_array,river_array,
                                        Card1_array_player2,Card2_array_player2,all_card_array_player2,pot_array_stage))
        y_player2 = Model.predict(model_player2,x=state_array_player2)
        action_player2 = get_action(y_player2)

        if action_player2 == "Fold":
            return "Player 1",stage_initial_potsize,0,0
        elif action_player2 == "Check/Call":
            return "",stage_initial_potsize,0,0
        else:
            stage_initial_potsize += 100
            pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
            state_array_player1 = np.stack((flop1_array, flop2_array, flop3_array,turn_array,river_array,
                                            Card1_array_player1,Card2_array_player1, all_card_array_player1,
                                            pot_array_stage))
            y_player1 = Model.predict(model_player1,x=state_array_player1)
            action_player1 = get_action_position2(y_player1)
            if action_player1 == "Fold":
                return "Player 2",stage_initial_potsize,0,100
            else:
                return "",stage_initial_potsize+100,100,100
    else:
        stage_initial_potsize += 100
        pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
        state_array_player2 = np.stack((flop1_array, flop2_array, flop3_array,turn_array,river_array,Card1_array_player2
                                        , Card2_array_player2, all_card_array_player2, pot_array_stage))
        y_player2 = Model.predict(model_player2,x=state_array_player2)
        action_player2 = get_action_position2(y_player2)
        if action_player2 == "Fold":
            return "Player 1",stage_initial_potsize,100,0
        else:
            return "", stage_initial_potsize+100,100,100


model_player1 = models.load_model("models\\heuristic.model")
model_player2 = models.load_model("models\\heuristic.model")

player1_bet = 0
player2_bet = 0
total_pot_size = 0
all_card_array_player1 = np.zeros((4,13))
all_card_array_player2 = np.zeros((4,13))
flop1_array,all_card_array_player1,all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(flop[0]),
                                                                                           all_card_array_player1,
                                                                                           all_card_array_player2)
flop2_array,all_card_array_player1,all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(flop[1]),
                                                                                           all_card_array_player1,
                                                                                           all_card_array_player2)
flop3_array,all_card_array_player1,all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(flop[2]),
                                                                                           all_card_array_player1,
                                                                                           all_card_array_player2)
turn_array = np.zeros((4,13))
river_array = np.zeros((4,13))
Card6_array_player1,all_card_array_player1 = convert_to_numpy_array_playercards(Card.int_to_str(player1_hand[0]),all_card_array_player1)
Card7_array_player1,all_card_array_player1 = convert_to_numpy_array_playercards(Card.int_to_str(player1_hand[0]),all_card_array_player1)

Card6_array_player2,all_card_array_player2 = convert_to_numpy_array_playercards(Card.int_to_str(player2_hand[0]),all_card_array_player2)
Card7_array_player2,all_card_array_player2 = convert_to_numpy_array_playercards(Card.int_to_str(player2_hand[0]),all_card_array_player2)


winner,pot_size,player1_new_bet,player2_new_bet = betting(model_player1,model_player2,flop1_array,flop2_array,flop3_array,
                                                          turn_array,river_array,Card6_array_player1,Card7_array_player1,
                                                          Card6_array_player2,Card7_array_player2,all_card_array_player1,
                                                          all_card_array_player2,total_pot_size)
total_pot_size += pot_size
player1_bet += player1_new_bet
player2_bet += player2_new_bet

if winner == "Player 1":
    print("Player 1 Wins!!",total_pot_size)
    print("Player 1 Gain!!",int(total_pot_size-player1_bet))

elif winner == "Player 2":
    print("Player 2 Wins!!",total_pot_size)
    print("Player 2 Gain!!",int(total_pot_size-player2_bet))

else:
    turn_array,all_card_array_player1,all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(turn),all_card_array_player1,all_card_array_player2)
    winner, pot_size, player1_new_bet, player2_new_bet = betting(model_player1, model_player2, flop1_array, flop2_array,
                                                                 flop3_array,
                                                                 turn_array, river_array, Card6_array_player1,
                                                                 Card7_array_player1,
                                                                 Card6_array_player2, Card7_array_player2,
                                                                 all_card_array_player1,
                                                                 all_card_array_player2, total_pot_size)
    total_pot_size += pot_size
    player1_bet += player1_new_bet
    player2_bet += player2_new_bet

    if winner == "Player 1":
        print("Player 1 Wins!!", total_pot_size)
        print("Player 1 Gain!!", int(total_pot_size - player1_bet))

    elif winner == "Player 2":
        print("Player 2 Wins!!", total_pot_size)
        print("Player 2 Gain!!", int(total_pot_size - player2_bet))

    else:
        river_array, all_card_array_player1, all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(river),
                                                                                            all_card_array_player1,
                                                                                            all_card_array_player2)
        winner, pot_size, player1_new_bet, player2_new_bet = betting(model_player1, model_player2, flop1_array,
                                                                     flop2_array,
                                                                     flop3_array,
                                                                     turn_array, river_array, Card6_array_player1,
                                                                     Card7_array_player1,
                                                                     Card6_array_player2, Card7_array_player2,
                                                                     all_card_array_player1,
                                                                     all_card_array_player2, total_pot_size)
        total_pot_size += pot_size
        player1_bet += player1_new_bet
        player2_bet += player2_new_bet

        if winner == "Player 1":
            print("Player 1 Wins!!", total_pot_size)
            print("Player 1 Gain!!", int(total_pot_size - player1_bet))

        elif winner == "Player 2":
            print("Player 2 Wins!!", total_pot_size)
            print("Player 2 Gain!!", int(total_pot_size - player2_bet))

        else:
            final_score_player1 = evaluator.get_rank_class(evaluator._seven(player1_complete_hand))
            final_score_player2 = evaluator.get_rank_class(evaluator._seven(player2_complete_hand))
            if final_score_player1 > final_score_player2:
                print("Player 1 Wins!!",total_pot_size)
                print("Player 1 Gain!!",int(total_pot_size - player1_bet))

            else:
                print("Player 2 Wins!!",total_pot_size)
                print("Player 2 Gain!!",int(total_pot_size-player2_bet))








