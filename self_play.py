import sys
from keras import models, Model
from treys import Deck, Evaluator, Card
import numpy as np
import pickle


def run(rounds, model):
    reward_table = dict()
    reward_count = dict()
    print("running self-play")
    round_number = 0
    print_progress(round_number, rounds, prefix='Progress:', suffix='Complete', bar_length=40)
    while round_number < rounds:
        deck = Deck()
        evaluator = Evaluator()
        flop = deck.draw(3)
        # for card in flop:
        #     print(Card.int_to_pretty_str(card))
        player1_hand = deck.draw(2)
        # for card in player1_hand:
        #     print(Card.int_to_pretty_str(card))
        player2_hand = deck.draw(3)
        turn = deck.draw(1)
        river = deck.draw(1)
        player1_complete_hand = flop + [turn] + [river] + player1_hand
        player2_complete_hand = flop + [turn] + [river] + player2_hand

        def get_action_position2(y_stage,other_player_action):
            if other_player_action == "Bet":
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
                return "Bet"


        def get_action(y_stage):
            index = 0
            max_gain_loss = y_stage[0][0]
            for i in range(1 ,len(y_stage[0])):
                if max_gain_loss < y_stage[0][i]:
                    index = i
                    max_gain_loss = y_stage[0][i]
            if index == 0:
                return "Fold"
            elif index == 1:
                return "Check/Call"
            else:
                return "Bet"


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


        def betting(model1_player, model2_player, flop1_array, flop2_array, flop3_array, turn_array, river_array, Card1_array_player1,
                    Card2_array_player1, Card1_array_player2, Card2_array_player2, all_card_array_player1, all_card_array_player2,
                    stage_initial_potsize):

            pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
            state_array_player1 = np.stack((flop1_array,flop2_array,flop3_array,turn_array,river_array,Card1_array_player1,
                                            Card2_array_player1,all_card_array_player1,pot_array_stage))
            state_array_player1 = np.expand_dims(state_array_player1,0)

            input_shape = [9, 17, 17]
            right_left_pad = input_shape[1] - state_array_player1.shape[2]
            left_pad = right_left_pad // 2
            right_pad = left_pad + (right_left_pad % 2)
            top_bottom_pad = input_shape[2] - state_array_player1.shape[3]
            top_pad = top_bottom_pad // 2
            bottom_pad = top_pad + (top_bottom_pad % 2)
            state_array_player1 = np.pad(state_array_player1, ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
            y_player1 = Model.predict(model1_player,x=state_array_player1)
            action_player1 = get_action(y_player1)
            if action_player1 == "Fold":
                return "Player 2",stage_initial_potsize,0,0,action_player1,""
            elif action_player1 == "Check/Call":
                state_array_player2 = np.stack((flop1_array,flop2_array,flop3_array,turn_array,river_array,
                                                Card1_array_player2,Card2_array_player2,all_card_array_player2,pot_array_stage))
                state_array_player2 = np.expand_dims(state_array_player2, 0)
                right_left_pad = input_shape[1] - state_array_player2.shape[2]
                left_pad = right_left_pad // 2
                right_pad = left_pad + (right_left_pad % 2)
                top_bottom_pad = input_shape[2] - state_array_player2.shape[3]
                top_pad = top_bottom_pad // 2
                bottom_pad = top_pad + (top_bottom_pad % 2)
                state_array_player2 = np.pad(state_array_player2,
                                             ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
                y_player2 = Model.predict(model2_player, x=state_array_player2)
                action_player2 = get_action(y_player2)

                if action_player2 == "Fold":
                    return "Player 1",stage_initial_potsize,0,0,action_player1,action_player2
                elif action_player2 == "Check/Call":
                    return "",stage_initial_potsize,0,0,action_player1,action_player2
                else:
                    stage_initial_potsize += 100
                    pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
                    state_array_player1 = np.stack((flop1_array, flop2_array, flop3_array,turn_array,river_array,
                                                    Card1_array_player1,Card2_array_player1, all_card_array_player1,
                                                    pot_array_stage))
                    state_array_player1 = np.expand_dims(state_array_player1, 0)
                    right_left_pad = input_shape[1] - state_array_player1.shape[2]
                    left_pad = right_left_pad // 2
                    right_pad = left_pad + (right_left_pad % 2)
                    top_bottom_pad = input_shape[2] - state_array_player1.shape[3]
                    top_pad = top_bottom_pad // 2
                    bottom_pad = top_pad + (top_bottom_pad % 2)
                    state_array_player1 = np.pad(state_array_player1,
                                                 ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)),
                                                 mode='constant')
                    y_player1 = Model.predict(model1_player, x=state_array_player1)
                    action_player1 = get_action_position2(y_player1,"Bet")
                    if action_player1 == "Fold":
                        return "Player 2",stage_initial_potsize,0,100,action_player1,action_player2
                    else:
                        return "",stage_initial_potsize+100,100,100,action_player1,action_player2
            else:
                stage_initial_potsize += 100
                pot_array_stage = convert_pot_to_numpy(stage_initial_potsize)
                state_array_player2 = np.stack((flop1_array, flop2_array, flop3_array,turn_array,river_array,Card1_array_player2
                                                , Card2_array_player2, all_card_array_player2, pot_array_stage))
                state_array_player2 = np.expand_dims(state_array_player2, 0)
                right_left_pad = input_shape[1] - state_array_player2.shape[2]
                left_pad = right_left_pad // 2
                right_pad = left_pad + (right_left_pad % 2)
                top_bottom_pad = input_shape[2] - state_array_player2.shape[3]
                top_pad = top_bottom_pad // 2
                bottom_pad = top_pad + (top_bottom_pad % 2)
                state_array_player2 = np.pad(state_array_player2,
                                             ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
                y_player2 = Model.predict(model2_player, x=state_array_player2)
                # print(y_player2)
                action_player2 = get_action_position2(y_player2,"Bet")
                if action_player2 == "Fold":
                    return "Player 1",stage_initial_potsize,100,0,action_player1,action_player2
                else:
                    return "", stage_initial_potsize+100,100,100,action_player1,action_player2


        model_player1 = model
        model_player2 = model

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
        Card7_array_player1,all_card_array_player1 = convert_to_numpy_array_playercards(Card.int_to_str(player1_hand[1]),all_card_array_player1)

        Card6_array_player2,all_card_array_player2 = convert_to_numpy_array_playercards(Card.int_to_str(player2_hand[0]),all_card_array_player2)
        Card7_array_player2,all_card_array_player2 = convert_to_numpy_array_playercards(Card.int_to_str(player2_hand[1]),all_card_array_player2)
        pot_array_stage1 = convert_pot_to_numpy(total_pot_size)
        state_array_player1_stage1 = np.stack((flop1_array,flop2_array,flop3_array,turn_array,river_array,Card6_array_player1,
                                            Card7_array_player1,all_card_array_player1,pot_array_stage1))

        hash_key_stage1 = pickle.dumps(state_array_player1_stage1)
        gain_loss_stage1 = np.zeros((3))
        gain_loss_stage1_count = np.zeros((3))
        winner,pot_size,player1_new_bet,player2_new_bet,player1_action_stage1,player2_action = betting(model_player1,model_player2,flop1_array,flop2_array,flop3_array,
                                                                  turn_array,river_array,Card6_array_player1,Card7_array_player1,
                                                                  Card6_array_player2,Card7_array_player2,all_card_array_player1,
                                                                  all_card_array_player2,total_pot_size)
        total_pot_size += pot_size
        player1_bet += player1_new_bet
        player2_bet += player2_new_bet

        if winner == "Player 1":
            # print("Player 1 Wins!!",total_pot_size)
            # print("Player 1 Gain!!",int(total_pot_size-player1_bet))
            if player1_action_stage1 == "Bet":
                if hash_key_stage1 in reward_table.keys():
                    reward_table[hash_key_stage1][2] += int(total_pot_size-player1_bet)
                    reward_count[hash_key_stage1][2] += 1
                else:
                    gain_loss_stage1[2] += int(total_pot_size-player1_bet)
                    gain_loss_stage1_count[2] += 1
            elif player1_action_stage1 == "Check/Call":
                if hash_key_stage1 in reward_table.keys():
                    reward_table[hash_key_stage1][1] += int(total_pot_size-player1_bet)
                    reward_count[hash_key_stage1][1] += 1
                else:
                    gain_loss_stage1[1] += int(total_pot_size-player1_bet)
                    gain_loss_stage1_count[1] += 1


        elif winner == "Player 2":
            # print("Player 2 Wins!!",total_pot_size)
            # print("Player 2 Gain!!",int(total_pot_size-player2_bet))
            if player1_action_stage1 == "Fold":
                if hash_key_stage1 in reward_table.keys():
                    reward_table[hash_key_stage1][0] = -1*int(player1_bet)
                    reward_count[hash_key_stage1][0] += 1
                else:
                    gain_loss_stage1[0] = -1*int(player1_bet)
                    gain_loss_stage1_count[0] = 1


        else:
            turn_array,all_card_array_player1,all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(turn),all_card_array_player1,all_card_array_player2)
            pot_array_stage2 = convert_pot_to_numpy(total_pot_size)
            state_array_player1_stage2 = np.stack(
                (flop1_array, flop2_array, flop3_array, turn_array, river_array, Card6_array_player1,
                 Card7_array_player1, all_card_array_player1, pot_array_stage2))

            hash_key_stage2 = pickle.dumps(state_array_player1_stage2)
            gain_loss_stage2 = np.zeros((3))
            gain_loss_stage2_count = np.zeros((3))
            winner, pot_size, player1_new_bet, player2_new_bet,player1_action_stage2,player2_action = betting(model_player1, model_player2, flop1_array, flop2_array,
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
                # print("Player 1 Wins!!", total_pot_size)
                # print("Player 1 Gain!!", int(total_pot_size - player1_bet))
                if player1_action_stage2 == "Bet":
                    if hash_key_stage2 in reward_table.keys():
                        reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                        reward_count[hash_key_stage2][2] += 1
                    else:
                        gain_loss_stage2[2] = int(total_pot_size - player1_bet)
                        gain_loss_stage2_count[2] = 1
                elif player1_action_stage2 == "Check/Call":
                    if hash_key_stage2 in reward_table.keys():
                        reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                        reward_count[hash_key_stage2][1] += 1
                    else:
                        gain_loss_stage2[1] = int(total_pot_size - player1_bet)
                        gain_loss_stage2_count[1] = 1
                if player1_action_stage1 == "Bet":
                    if hash_key_stage1 in reward_table.keys():
                        reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                        reward_count[hash_key_stage1][2] += 1
                    else:
                        gain_loss_stage1[2] = int(total_pot_size - player1_bet)
                        gain_loss_stage1_count[2] = 1
                elif player1_action_stage1 == "Check/Call":
                    if hash_key_stage1 in reward_table.keys():
                        reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                        reward_count[hash_key_stage1][1] += 1
                    else:
                        gain_loss_stage1[1] = int(total_pot_size - player1_bet)
                        gain_loss_stage1_count[1] = 1

            elif winner == "Player 2":
                # print("Player 2 Wins!!", total_pot_size)
                # print("Player 2 Gain!!", int(total_pot_size - player2_bet))
                if player1_action_stage2 == "Fold":
                    if hash_key_stage2 in reward_table.keys():
                        reward_table[hash_key_stage2][0] = -1 * int(player1_bet)
                        reward_count[hash_key_stage2][0] += 1
                    else:
                        gain_loss_stage2[0] = -1 * int(player1_bet)
                        gain_loss_stage2_count[0] += 1
                if player1_action_stage1 == "Bet":
                    if hash_key_stage1 in reward_table.keys():
                        reward_table[hash_key_stage1][2] += -1 * int(player1_bet)
                        reward_count[hash_key_stage1][2] += 1
                    else:
                        gain_loss_stage1[2] = -1 * int(player1_bet)
                        gain_loss_stage1_count[2] = 1
                elif player1_action_stage1 == "Check/Call":
                    if hash_key_stage1 in reward_table.keys():
                        reward_table[hash_key_stage1][1] += -1 * int(player1_bet)
                        reward_count[hash_key_stage1][1] += 1
                    else:
                        gain_loss_stage1[1] = -1 * int(player1_bet)
                        gain_loss_stage1_count[1] = 1

            else:
                river_array, all_card_array_player1, all_card_array_player2 = convert_to_numpy_array(Card.int_to_str(river),
                                                                                                    all_card_array_player1,
                                                                                                    all_card_array_player2)
                winner, pot_size, player1_new_bet, player2_new_bet,player1_action_stage3,player2_action = betting(model_player1, model_player2, flop1_array,
                                                                             flop2_array,
                                                                             flop3_array,
                                                                             turn_array, river_array, Card6_array_player1,
                                                                             Card7_array_player1,
                                                                             Card6_array_player2, Card7_array_player2,
                                                                             all_card_array_player1,
                                                                             all_card_array_player2, total_pot_size)
                pot_array_stage3 = convert_pot_to_numpy(total_pot_size)
                state_array_player1_stage3 = np.stack(
                    (flop1_array, flop2_array, flop3_array, turn_array, river_array, Card6_array_player1,
                     Card7_array_player1, all_card_array_player1, pot_array_stage3))

                hash_key_stage3 = pickle.dumps(state_array_player1_stage3)
                gain_loss_stage3 = np.zeros((3))
                gain_loss_stage3_count = np.zeros((3))

                total_pot_size += pot_size
                player1_bet += player1_new_bet
                player2_bet += player2_new_bet

                if winner == "Player 1":
                    # print("Player 1 Wins!!", total_pot_size)
                    # print("Player 1 Gain!!", int(total_pot_size - player1_bet))
                    if player1_action_stage3 == "Bet":
                        if hash_key_stage3 in reward_table.keys():
                            reward_table[hash_key_stage3][2] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage3][2] += 1
                        else:
                            gain_loss_stage3[2] = int(total_pot_size - player1_bet)
                            gain_loss_stage3_count[2] = 1
                    elif player1_action_stage3 == "Check/Call":
                        if hash_key_stage3 in reward_table.keys():
                            reward_table[hash_key_stage3][1] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage3][1] += 1
                        else:
                            gain_loss_stage3[1] = int(total_pot_size - player1_bet)
                            gain_loss_stage3_count[1] = 1
                    if player1_action_stage1 == "Bet":
                        if hash_key_stage1 in reward_table.keys():
                            reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage1][2] += 1
                        else:
                            gain_loss_stage1[2] = int(total_pot_size - player1_bet)
                            gain_loss_stage1_count[2] = 1
                    elif player1_action_stage1 == "Check/Call":
                        if hash_key_stage1 in reward_table.keys():
                            reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage1][1] += 1
                        else:
                            gain_loss_stage1[1] = int(total_pot_size - player1_bet)
                            gain_loss_stage1_count[1] = 1
                    if player1_action_stage2 == "Bet":
                        if hash_key_stage2 in reward_table.keys():
                            reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage2][2] += 1
                        else:
                            gain_loss_stage2[2] = int(total_pot_size - player1_bet)
                            gain_loss_stage2_count[2] = 1
                    elif player1_action_stage2 == "Check/Call":
                        if hash_key_stage2 in reward_table.keys():
                            reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                            reward_count[hash_key_stage2][1] += 1
                        else:
                            gain_loss_stage2[1] = int(total_pot_size - player1_bet)
                            gain_loss_stage2_count[1] = 1

                elif winner == "Player 2":
                    # print("Player 2 Wins!!", total_pot_size)
                    # print("Player 2 Gain!!", int(total_pot_size - player2_bet))
                    if player1_action_stage3 == "Fold":
                        if hash_key_stage3 in reward_table.keys():
                            reward_table[hash_key_stage3][0] = -1 * int(player1_bet)
                            reward_count[hash_key_stage3][0] += 1
                        else:
                            gain_loss_stage3[0] = -1 * int(player1_bet)
                            gain_loss_stage3_count[0] += 1

                    if player1_action_stage1 == "Bet":
                        if hash_key_stage1 in reward_table.keys():
                            reward_table[hash_key_stage1][2] += -1 * int(player1_bet)
                            reward_count[hash_key_stage1][2] += 1
                        else:
                            gain_loss_stage1[2] = -1 * int(player1_bet)
                            gain_loss_stage1_count[2] = 1
                    elif player1_action_stage1 == "Check/Call":
                        if hash_key_stage1 in reward_table.keys():
                            reward_table[hash_key_stage1][1] += -1 * int(player1_bet)
                            reward_count[hash_key_stage1][1] += 1
                        else:
                            gain_loss_stage1[1] = -1 * int(player1_bet)
                            gain_loss_stage1_count[1] = 1

                    if player1_action_stage2 == "Bet":
                        if hash_key_stage2 in reward_table.keys():
                            reward_table[hash_key_stage2][2] += -1 * int(player1_bet)
                            reward_count[hash_key_stage2][2] += 1
                        else:
                            gain_loss_stage2[2] = -1 * int(player1_bet)
                            gain_loss_stage2_count[2] = 1
                    elif player1_action_stage2 == "Check/Call":
                        if hash_key_stage2 in reward_table.keys():
                            reward_table[hash_key_stage2][1] += -1 * int(player1_bet)
                            reward_count[hash_key_stage2][1] += 1
                        else:
                            gain_loss_stage2[1] = -1 * int(player1_bet)
                            gain_loss_stage2_count[1] = 1

                else:
                    final_score_player1 = evaluator.get_rank_class(evaluator._seven(player1_complete_hand))
                    final_score_player2 = evaluator.get_rank_class(evaluator._seven(player2_complete_hand))
                    if final_score_player1 < final_score_player2:
                        # print("Player 1 Wins!!",total_pot_size)
                        # print("Player 1 Gain!!",int(total_pot_size - player1_bet))
                        if player1_action_stage3 == "Bet":
                            if hash_key_stage3 in reward_table.keys():
                                reward_table[hash_key_stage3][2] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage3][2] += 1
                            else:
                                gain_loss_stage3[2] = int(total_pot_size - player1_bet)
                                gain_loss_stage3_count[2] = 1
                        elif player1_action_stage3 == "Check/Call":
                            if hash_key_stage3 in reward_table.keys():
                                reward_table[hash_key_stage3][1] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage3][1] += 1
                            else:
                                gain_loss_stage3[1] = int(total_pot_size - player1_bet)
                                gain_loss_stage3_count[1] = 1
                        if player1_action_stage1 == "Bet":
                            if hash_key_stage1 in reward_table.keys():
                                reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage1][2] += 1
                            else:
                                gain_loss_stage1[2] = int(total_pot_size - player1_bet)
                                gain_loss_stage1_count[2] = 1
                        elif player1_action_stage1 == "Check/Call":
                            if hash_key_stage1 in reward_table.keys():
                                reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage1][1] += 1
                            else:
                                gain_loss_stage1[1] = int(total_pot_size - player1_bet)
                                gain_loss_stage1_count[1] = 1
                        if player1_action_stage2 == "Bet":
                            if hash_key_stage2 in reward_table.keys():
                                reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage2][2] += 1
                            else:
                                gain_loss_stage2[2] = int(total_pot_size - player1_bet)
                                gain_loss_stage2_count[2] = 1
                        elif player1_action_stage2 == "Check/Call":
                            if hash_key_stage2 in reward_table.keys():
                                reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                                reward_count[hash_key_stage2][1] += 1
                            else:
                                gain_loss_stage2[1] = int(total_pot_size - player1_bet)
                                gain_loss_stage2_count[1] = 1

                    else:
                        # print("Player 2 Wins!!",total_pot_size)
                        # print("Player 2 Gain!!",int(total_pot_size-player2_bet))
                        # print("Player 2 Wins!!", total_pot_size)
                        # print("Player 2 Gain!!", int(total_pot_size - player2_bet))
                        if player1_action_stage3 == "Fold":
                            if hash_key_stage3 in reward_table.keys():
                                reward_table[hash_key_stage3][0] = -1 * int(player1_bet)
                                reward_count[hash_key_stage3][0] += 1
                            else:
                                gain_loss_stage3[0] = -1 * int(player1_bet)
                                gain_loss_stage3_count[0] += 1

                        if player1_action_stage1 == "Bet":
                            if hash_key_stage1 in reward_table.keys():
                                reward_table[hash_key_stage1][2] += -1 * int(player1_bet)
                                reward_count[hash_key_stage1][2] += 1
                            else:
                                gain_loss_stage1[2] = -1 * int(player1_bet)
                                gain_loss_stage1_count[2] = 1
                        elif player1_action_stage1 == "Check/Call":
                            if hash_key_stage1 in reward_table.keys():
                                reward_table[hash_key_stage1][1] += -1 * int(player1_bet)
                                reward_count[hash_key_stage1][1] += 1
                            else:
                                gain_loss_stage1[1] = -1 * int(player1_bet)
                                gain_loss_stage1_count[1] = 1

                        if player1_action_stage2 == "Bet":
                            if hash_key_stage2 in reward_table.keys():
                                reward_table[hash_key_stage2][2] += -1 * int(player1_bet)
                                reward_count[hash_key_stage2][2] += 1
                            else:
                                gain_loss_stage2[2] = -1 * int(player1_bet)
                                gain_loss_stage2_count[2] = 1
                        elif player1_action_stage2 == "Check/Call":
                            if hash_key_stage2 in reward_table.keys():
                                reward_table[hash_key_stage2][1] += -1 * int(player1_bet)
                                reward_count[hash_key_stage2][1] += 1
                            else:
                                gain_loss_stage2[1] = -1 * int(player1_bet)
                                gain_loss_stage2_count[1] = 1

                if hash_key_stage3 not in reward_table.keys():
                    reward_table[hash_key_stage3] = gain_loss_stage3
                    reward_count[hash_key_stage3] = gain_loss_stage3_count

            if hash_key_stage2 not in reward_table.keys():
                reward_table[hash_key_stage2] = gain_loss_stage2
                reward_count[hash_key_stage2] = gain_loss_stage2_count

        if hash_key_stage1 not in reward_table.keys():
            reward_table[hash_key_stage1] = gain_loss_stage1
            reward_count[hash_key_stage1] = gain_loss_stage1_count
        round_number += 1
        print_progress(round_number, rounds, prefix='Progress:', suffix='Complete', bar_length=40)

    for key in reward_table.keys():
        count_fold = reward_count[key][0]
        count_check = reward_count[key][1]
        count_bet = reward_count[key][2]
        if count_fold!=0:
            reward_table[key][0] = reward_table[key][0]/count_fold
        if count_check!=0:
            reward_table[key][1] = reward_table[key][1]/count_check
        if count_bet!=0:
            reward_table[key][2] = reward_table[key][2]/count_bet
    return reward_table

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

