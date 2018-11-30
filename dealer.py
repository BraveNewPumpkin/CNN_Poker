from player import Player
from treys import Deck, Evaluator, Card
import numpy as np
import pickle

def run(rounds):
  reward_table = dict()
  reward_count = dict()
  i=0
  while i < rounds:
      deck = Deck()
      Player_1 = Player()
      Player_2 = Player()
      flop = deck.draw(3)
      print("Flop Cards")
      for cards in flop:
          print(Card.int_to_pretty_str(cards))
      player1_hand = deck.draw(2)
      print("Player 1 Cards")
      for cards in player1_hand:
          print(Card.int_to_pretty_str(cards))
      player2_hand = deck.draw(2)
      print("Player 2 Cards")
      for cards in player2_hand:
          print(Card.int_to_pretty_str(cards))
      evaluator = Evaluator()
      all_card_array_stage3=np.zeros((4,13))
      all_card_array_stage1 = np.zeros((4,13))
      all_card_array_stage2 = np.zeros((4,13))


      def convert_pot_to_numpy(total_pot):
          pot_array = np.zeros((4, 13))
          number_of_chips = int(total_pot/25)
          if number_of_chips > 13:
              pot_array[1] = 1
              left_over_chips = number_of_chips-13
              for i in range(0,left_over_chips):
                  pot_array[2][i] = 1
          else:
              for i in range(0,number_of_chips):
                  pot_array[1][i] = 1
          return pot_array


      def convert_to_numpy_array(Card_str,all_card_array):
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
              index_1 = int(Card_str[0])-2

          if Card_str[1] == "s":
              index_2 = 0
          elif Card_str[1] == "c":
              index_2 = 1
          elif Card_str[1] == "h":
              index_2 = 2
          else:
              index_2 = 3

          new_card_array = np.zeros((4,13))
          new_card_array[index_2][index_1] = 1
          all_card_array[index_2][index_1]=1
          return new_card_array,all_card_array


      def get_possible_actions(player_action):
          if player_action == "Check/Call":
              return ["Fold","Bet/Raise"]
          elif player_action == "Bet/Raise":
              return ["Fold","Check/Call"]
          else:
              return ["Fold","Bet/Raise","Check/Call"]


      def betting(player1_turn_rank, player2_turn_rank, all_possible_actions):
          action_1, player1_new_bet = Player_1.make_bets(player1_turn_rank, all_possible_actions,0)
          print("Player 1",action_1)
          if action_1 == "Fold":
              return "Player2", 0, 0, 0,action_1
          elif action_1 == "Bet/Raise":
              possible_actions = get_possible_actions(action_1)
              action_2,player2_new_bet = Player_2.make_bets(player2_turn_rank, possible_actions,player1_new_bet)
              print("Player 2", action_2)
              if action_2 == "Fold":
                  return "Player1", player1_new_bet, 0, player1_new_bet,action_1
              else:
                  return "", player1_new_bet, player1_new_bet, 2*player1_new_bet,action_1
          else:
              action_2,player2_new_bet = Player_2.make_bets(player2_turn_rank, all_possible_actions,0)
              print("Player 2", action_2)
              if action_2 == "Fold":
                  return "Player1", 0, 0, 0,action_1
              elif action_2 == "Check/Call":
                  return "", 0, 0, 0,action_1
              else:
                  possible_actions = get_possible_actions(action_2)
                  action_1,player1_new_bet = Player_1.make_bets(player1_turn_rank,possible_actions,player2_new_bet)
                  print("Player 1", action_1)
                  if action_1 == "Fold":
                      return "Player2", 0, player2_new_bet, player2_new_bet,action_1
                  else:
                      return "", player2_new_bet, player2_new_bet, 2*player2_new_bet,action_1


      total_pot_size = 0
      player1_bet = 0
      player2_bet = 0

      player1_turn1 = flop+player1_hand
      player2_turn1 = flop+player2_hand

      player1_turn1_rank = evaluator.class_to_string(evaluator.get_rank_class(evaluator._five(player1_turn1)))
      player2_turn1_rank = evaluator.class_to_string(evaluator.get_rank_class(evaluator._five(player2_turn1)))

      all_possible_action = ["Fold","Bet/Raise","Check/Call"]

      winner, player1_round1_bet, player2_round1_bet, round1_pot_size,player1_action_stage1 = betting(player1_turn1_rank,player2_turn1_rank, all_possible_action)
      player1_bet += player1_round1_bet
      player2_bet  += player2_round1_bet
      total_pot_size += round1_pot_size
      Card1_stage1_array,all_card_array_stage1 = convert_to_numpy_array(Card.int_to_str(flop[0]),all_card_array_stage1)
      Card2_stage1_array,all_card_array_stage1 = convert_to_numpy_array(Card.int_to_str(flop[1]),all_card_array_stage1)
      Card3_stage1_array,all_card_array_stage1 = convert_to_numpy_array(Card.int_to_str(flop[2]),all_card_array_stage1)
      Card4_stage1_array = np.zeros((4,13))
      Card5_stage1_array = np.zeros((4,13))
      Card6_stage1_array,all_card_array_stage1 = convert_to_numpy_array(Card.int_to_str(player1_hand[0]),all_card_array_stage1)
      Card7_stage1_array,all_card_array_stage1 = convert_to_numpy_array(Card.int_to_str(player1_hand[1]),all_card_array_stage1)

      pot_array_stage1 = convert_pot_to_numpy(total_pot_size)
      state_array_stage1 = np.stack(
          (Card1_stage1_array, Card2_stage1_array, Card3_stage1_array, Card4_stage1_array, Card5_stage1_array,
           Card6_stage1_array,Card7_stage1_array,all_card_array_stage1,pot_array_stage1))
      hash_key_stage1 = pickle.dumps(state_array_stage1)
      gain_loss_table_stage1 = np.zeros((3))
      gain_loss_count_stage1 = np.zeros((3))

      if winner == "Player1":
          print(player1_action_stage1)
          if player1_action_stage1 == "Bet/Raise":
              if hash_key_stage1 in reward_table.keys():
                  reward_table[hash_key_stage1][2] += int(total_pot_size-player1_bet)
                  reward_count[hash_key_stage1][2] += 1
              else:
                  gain_loss_count_stage1[2] += 1
                  gain_loss_table_stage1[2] += int(total_pot_size-player1_bet)

          elif player1_action_stage1 == "Check/Call":
              if hash_key_stage1 in reward_table.keys():
                  reward_table[hash_key_stage1][1] += int(total_pot_size-player1_bet)
                  reward_count[hash_key_stage1][1] += 1
              else:
                  gain_loss_count_stage1[1] += 1
                  gain_loss_table_stage1[1] += int(total_pot_size-player1_bet)

          print("Player 1 Wins: ",total_pot_size)
          print("Player 1 Gain: ", total_pot_size-player1_bet)
      elif winner == "Player2":
          if hash_key_stage1 in reward_table.keys():
              if player1_action_stage1 == "Fold":
                  reward_table[hash_key_stage1][0] += -1 * int(player1_bet)
                  reward_count[hash_key_stage1][0] += 1
          else:
              if player1_action_stage1 == "Fold":
                  gain_loss_table_stage1[0] = -1 * int(player1_bet)
                  gain_loss_count_stage1[0] += 1
          print("Player 2 Wins: ",total_pot_size)
          print("Player 2 Gain: ", total_pot_size-player2_bet)
      else:
          turn = deck.draw(1)
          print("Turn ",Card.int_to_pretty_str(turn))
          player1_turn2 = player1_turn1 + [turn]
          player2_turn2 = player2_turn1 + [turn]

          player1_turn2_rank =  evaluator.class_to_string(evaluator.get_rank_class(evaluator._six(player1_turn2)))
          player2_turn2_rank = evaluator.class_to_string(evaluator.get_rank_class(evaluator._six(player2_turn2)))

          winner,player1_round2_bet,player2_round2_bet,round2_pot_size,player1_action_stage2 = betting(player1_turn2_rank,player2_turn2_rank,all_possible_action)
          player1_bet += player1_round2_bet
          player2_bet += player2_round2_bet
          total_pot_size += round2_pot_size
          Card1_stage2_array, all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(flop[0]),
                                                                             all_card_array_stage2)
          Card2_stage2_array, all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(flop[1]),
                                                                             all_card_array_stage2)
          Card3_stage2_array, all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(flop[2]),
                                                                             all_card_array_stage2)
          Card4_stage2_array,all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(turn),all_card_array_stage2)
          Card5_stage2_array = np.zeros((4,13))
          Card6_stage2_array, all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(player1_hand[0]),
                                                                             all_card_array_stage2)
          Card7_stage2_array, all_card_array_stage2 = convert_to_numpy_array(Card.int_to_str(player1_hand[1]),
                                                                             all_card_array_stage2)

          pot_array_stage2 = convert_pot_to_numpy(total_pot_size)
          state_array_stage2 = np.stack(
              (Card1_stage2_array, Card2_stage2_array, Card3_stage2_array, Card4_stage2_array, Card5_stage2_array,
               Card6_stage2_array,Card7_stage2_array,all_card_array_stage2, pot_array_stage2))
          hash_key_stage2 = pickle.dumps(state_array_stage2)
          gain_loss_table_stage2 = np.zeros((3))
          gain_loss_count_stage2 = np.zeros((3))
          if winner == "Player1":
              print(player1_action_stage2)
              if player1_action_stage2 == "Bet/Raise":
                  if hash_key_stage2 in reward_table.keys():
                      reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                      reward_count[hash_key_stage2][2] += 1
                  else:
                      gain_loss_count_stage2[2] += 1
                      gain_loss_table_stage2[2] += int(total_pot_size - player1_bet)

              elif player1_action_stage2 == "Check/Call":
                  if hash_key_stage2 in reward_table.keys():
                      reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                      reward_count[hash_key_stage2][1] += 1
                  else:
                      gain_loss_count_stage2[1] += 1
                      gain_loss_table_stage2[1] += int(total_pot_size - player1_bet)
              if player1_action_stage1 == "Bet/Raise":
                  if hash_key_stage1 in reward_table.keys():
                      reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                      reward_count[hash_key_stage1][2] += 1
                  else:
                      gain_loss_count_stage1[2] += 1
                      gain_loss_table_stage1[2] += int(total_pot_size - player1_bet)

              elif player1_action_stage1 == "Check/Call":
                  if hash_key_stage1 in reward_table.keys():
                      reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                      reward_count[hash_key_stage1][1] += 1
                  else:
                      gain_loss_count_stage1[1] += 1
                      gain_loss_table_stage1[1] += int(total_pot_size - player1_bet)
              print("Player 1 Wins: ",total_pot_size)
              print("Player 1 Gain: ",total_pot_size-player1_bet)
          elif winner == "Player2":
              if hash_key_stage2 in reward_table.keys():
                  if player1_action_stage2 == "Fold":
                      reward_table[hash_key_stage2][0] += -1 * int(player1_bet)
                      reward_count[hash_key_stage2][0] += 1
              else:
                  if player1_action_stage2 == "Fold":
                      gain_loss_table_stage2[0] = -1 * int(player1_bet)
                      gain_loss_count_stage2[0] += 1
              if player1_action_stage1 == "Bet/Raise":
                  if hash_key_stage1 in reward_table.keys():
                      reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                      reward_count[hash_key_stage1][2] += 1
                  else:
                      gain_loss_count_stage1[2] += 1
                      gain_loss_table_stage1[2] += int(total_pot_size - player1_bet)

              elif player1_action_stage1 == "Check/Call":
                  if hash_key_stage1 in reward_table.keys():
                      reward_table[hash_key_stage1][1] += -1*int(player1_bet)
                      reward_count[hash_key_stage1][1] += 1
                  else:
                      gain_loss_count_stage1[1] += 1
                      gain_loss_table_stage1[1] += -1*int(player1_bet)
              print("Player 2 Wins: ",total_pot_size)
              print("Player 2 Gain: ",total_pot_size-player2_bet)
          else:
              river = deck.draw(1)
              print("River ",Card.int_to_pretty_str(river))
              player1_turn3 = player1_turn2 + [river]
              player2_turn3 = player2_turn2 + [river]

              #Creating state array
              Card1 = Card.int_to_str(flop[0])
              Card2 = Card.int_to_str(flop[1])
              Card3 = Card.int_to_str(flop[2])
              Card4 = Card.int_to_str(turn)
              Card5 = Card.int_to_str(river)
              Card6 = Card.int_to_str(player1_hand[0])
              Card7 = Card.int_to_str(player1_hand[1])
              Card1_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card1,all_card_array_stage3)
              Card2_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card2,all_card_array_stage3)
              Card3_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card3,all_card_array_stage3)
              Card4_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card4,all_card_array_stage3)
              Card5_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card5,all_card_array_stage3)
              Card6_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card6,all_card_array_stage3)
              Card7_stage3_array,all_card_array_stage3 = convert_to_numpy_array(Card7,all_card_array_stage3)

              player1_turn3_rank = evaluator.class_to_string(evaluator.get_rank_class(evaluator._seven(player1_turn3)))
              player2_turn3_rank = evaluator.class_to_string(evaluator.get_rank_class(evaluator._seven(player2_turn3)))

              winner,player1_round3_bet,player2_round3_bet,round3_pot_size,player1_action = betting(player1_turn3_rank,player2_turn3_rank,all_possible_action)
              player1_bet += player1_round3_bet
              player2_bet += player2_round3_bet
              total_pot_size += round3_pot_size
              total_pot_array = convert_pot_to_numpy(total_pot_size)
              state_array_stage3 = np.stack(
                  (Card1_stage3_array, Card2_stage3_array, Card3_stage3_array, Card4_stage3_array,
                   Card5_stage3_array, Card6_stage3_array, Card7_stage3_array,
                   all_card_array_stage3,total_pot_array))
              # print(state_array)
              hash_key = pickle.dumps(state_array_stage3)
              gain_loss_table_stage3 = np.zeros((3))
              gain_loss_count_stage3 = np.zeros((3))
              i += 1
              if winner == "Player1":
                  print(player1_action)
                  if hash_key in reward_table.keys():

                      if player1_action == "Check/Call":
                          reward_table[hash_key][1] += int(total_pot_size-player1_bet)
                          reward_count[hash_key][1]+=1
                      elif player1_action == "Bet/Raise":
                          reward_table[hash_key][2] += int(total_pot_size-player1_bet)
                          reward_count[hash_key][2]+=1
                  else:
                      if player1_action == "Check/Call":
                          gain_loss_table_stage3[1] = int(total_pot_size - player1_bet)
                          gain_loss_count_stage3[1] += 1
                      elif player1_action == "Bet/Raise":
                          gain_loss_table_stage3[2] = int(total_pot_size - player1_bet)
                          gain_loss_count_stage3[2] += 1

                  if player1_action_stage2 == "Bet/Raise":
                      if hash_key_stage2 in reward_table.keys():
                          reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                          reward_count[hash_key_stage2][2] += 1
                      else:
                          gain_loss_count_stage2[2] += 1
                          gain_loss_table_stage2[2] += int(total_pot_size - player1_bet)

                  elif player1_action_stage2 == "Check/Call":
                      if hash_key_stage2 in reward_table.keys():
                          reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                          reward_count[hash_key_stage2][1] += 1
                      else:
                          gain_loss_count_stage2[1] += 1
                          gain_loss_table_stage2[1] += int(total_pot_size - player1_bet)

                  if player1_action_stage1 == "Bet/Raise":
                      if hash_key_stage1 in reward_table.keys():
                          reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                          reward_count[hash_key_stage1][2] += 1
                      else:
                          gain_loss_count_stage1[2] += 1
                          gain_loss_table_stage1[2] += int(total_pot_size - player1_bet)

                  elif player1_action_stage1 == "Check/Call":
                      if hash_key_stage1 in reward_table.keys():
                          reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                          reward_count[hash_key_stage1][1] += 1
                      else:
                          gain_loss_count_stage1[1] += 1
                          gain_loss_table_stage1[1] += int(total_pot_size - player1_bet)


                  print("Player 1 Wins: ",total_pot_size)
                  print("Player 1 Gain: ",total_pot_size-player1_bet)
              elif winner == "Player2":
                  if hash_key in reward_table.keys():
                      if player1_action == "Fold":
                          reward_table[hash_key][0] += -1*int(player1_bet)
                          reward_count[hash_key][0] += 1
                  else:
                      if player1_action == "Fold":
                          gain_loss_table_stage3[0] = -1*int(player1_bet)
                          gain_loss_count_stage3[0] += 1
                  if player1_action_stage2 == "Bet/Raise":
                      if hash_key_stage2 in reward_table.keys():
                          reward_table[hash_key_stage2][2] += -1*int(player1_bet)
                          reward_count[hash_key_stage2][2] += 1
                      else:
                          gain_loss_count_stage2[2] += 1
                          gain_loss_table_stage2[2] += -1*int(player1_bet)

                  elif player1_action_stage2 == "Check/Call":
                      if hash_key_stage2 in reward_table.keys():
                          reward_table[hash_key_stage2][1] += -1*int(player1_bet)
                          reward_count[hash_key_stage2][1] += 1
                      else:
                          gain_loss_count_stage2[1] += 1
                          gain_loss_table_stage2[1] += -1*int(player1_bet)

                  if player1_action_stage1 == "Bet/Raise":
                      if hash_key_stage1 in reward_table.keys():
                          reward_table[hash_key_stage1][2] += -1*int(player1_bet)
                          reward_count[hash_key_stage1][2] += 1
                      else:
                          gain_loss_count_stage1[2] += 1
                          gain_loss_table_stage1[2] += -1*int(player1_bet)

                  elif player1_action_stage1 == "Check/Call":
                      if hash_key_stage1 in reward_table.keys():
                          reward_table[hash_key_stage1][1] += -1*int(player1_bet)
                          reward_count[hash_key_stage1][1] += 1
                      else:
                          gain_loss_count_stage1[1] += 1
                          gain_loss_table_stage1[1] += -1*int(player1_bet)

                  print("Player 2 Wins: ",total_pot_size)
                  print("Player 2 Gain: ",total_pot_size-player2_bet)
              else:
                  final_score_player1 = evaluator.get_rank_class(evaluator._seven(player1_turn3))
                  final_score_player2 = evaluator.get_rank_class(evaluator._seven(player2_turn3))
                  if final_score_player1 > final_score_player2:
                      print(player1_action)
                      if hash_key in reward_table.keys():
                          if player1_action == "Check/Call":
                              reward_table[hash_key][1] += int(total_pot_size-player1_bet)
                              reward_count[hash_key][1] +=1
                          elif player1_action == "Bet/Raise":
                              reward_table[hash_key][2] += int(total_pot_size-player1_bet)
                              reward_count[hash_key][2] +=1
                      else:
                          if player1_action == "Check/Call":
                              gain_loss_table_stage3[1] = int(total_pot_size - player1_bet)
                              gain_loss_count_stage3[1] +=1
                          elif player1_action == "Bet/Raise":
                              gain_loss_table_stage3[2] = int(total_pot_size - player1_bet)
                              gain_loss_count_stage3[2] += 1
                      if player1_action_stage2 == "Bet/Raise":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage2][2] += int(total_pot_size - player1_bet)
                              reward_count[hash_key_stage2][2] += 1
                          else:
                              gain_loss_count_stage2[2] += 1
                              gain_loss_table_stage2[2] += int(total_pot_size - player1_bet)

                      elif player1_action_stage2 == "Check/Call":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage2][1] += int(total_pot_size - player1_bet)
                              reward_count[hash_key_stage2][1] += 1
                          else:
                              gain_loss_count_stage2[1] += 1
                              gain_loss_table_stage2[1] += int(total_pot_size - player1_bet)

                      if player1_action_stage1 == "Bet/Raise":
                          if hash_key_stage1 in reward_table.keys():
                              reward_table[hash_key_stage1][2] += int(total_pot_size - player1_bet)
                              reward_count[hash_key_stage1][2] += 1
                          else:
                              gain_loss_count_stage1[2] += 1
                              gain_loss_table_stage1[2] += int(total_pot_size - player1_bet)

                      elif player1_action_stage1 == "Check/Call":
                          if hash_key_stage1 in reward_table.keys():
                              reward_table[hash_key_stage1][1] += int(total_pot_size - player1_bet)
                              reward_count[hash_key_stage1][1] += 1
                          else:
                              gain_loss_count_stage1[1] += 1
                              gain_loss_table_stage1[1] += int(total_pot_size - player1_bet)
                      print("Player 1 Wins: ",total_pot_size)
                      print("Player 1 Gain: ",total_pot_size-player1_bet)
                  else:
                      if hash_key in reward_table.keys():
                          if player1_action == "Check/Call":
                              reward_table[hash_key][1] += -1*int(player1_bet)
                              reward_count[hash_key][1] += 1
                          elif player1_action == "Bet/Raise":
                              reward_table[hash_key][2] += -1*int(player1_bet)
                              reward_count[hash_key][2] += 1
                      else:
                          if player1_action == "Check/Call":
                              gain_loss_table_stage3[1] = -1*int(player1_bet)
                              gain_loss_count_stage3[1] += 1
                          elif player1_action == "Bet/Raise":
                              gain_loss_table_stage3[2] = -1*int(player1_bet)
                              gain_loss_count_stage3[2] += 1

                      if player1_action_stage2 == "Bet/Raise":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage2][2] += -1 * int(player1_bet)
                              reward_count[hash_key_stage2][2] += 1
                          else:
                              gain_loss_count_stage2[2] += 1
                              gain_loss_table_stage2[2] += -1 * int(player1_bet)

                      elif player1_action_stage2 == "Check/Call":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage2][1] += -1 * int(player1_bet)
                              reward_count[hash_key_stage2][1] += 1
                          else:
                              gain_loss_count_stage2[1] += 1
                              gain_loss_table_stage2[1] += -1 * int(player1_bet)

                      if player1_action_stage1 == "Bet/Raise":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage1][2] += -1 * int(player1_bet)
                              reward_count[hash_key_stage1][2] += 1
                          else:
                              gain_loss_count_stage1[2] += 1
                              gain_loss_table_stage1[2] += -1 * int(player1_bet)

                      elif player1_action_stage1 == "Check/Call":
                          if hash_key_stage2 in reward_table.keys():
                              reward_table[hash_key_stage1][1] += -1 * int(player1_bet)
                              reward_count[hash_key_stage1][1] += 1
                          else:
                              gain_loss_count_stage1[1] += 1
                              gain_loss_table_stage1[1] += -1 * int(player1_bet)
                      print("Player 2 Wins:  ",total_pot_size)
                      print("Player 2 Gain: ",total_pot_size-player2_bet)

              if hash_key not in reward_table.keys():
                  reward_table[hash_key] = gain_loss_table_stage3
                  reward_count[hash_key] = gain_loss_count_stage3

          if hash_key_stage2 not in reward_table.keys():
              reward_table[hash_key_stage2] = gain_loss_table_stage2
              reward_count[hash_key_stage2] = gain_loss_count_stage2

      if hash_key_stage1 not in reward_table.keys():
          reward_table[hash_key_stage1] = gain_loss_table_stage1
          reward_count[hash_key_stage1] = gain_loss_count_stage1



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













