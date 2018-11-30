from keras import models,Model
from treys import Deck, Evaluator, Card
import numpy as np

deck = Deck()
flop = deck.draw(3)
player1_hand = deck.draw(2)
player2_hand = deck.draw(3)
turn = deck.draw(1)
river = deck.draw(1)


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


def convert_to_numpy_array(Card_str, all_card_array):
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


