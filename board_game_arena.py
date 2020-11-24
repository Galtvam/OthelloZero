import os
import time

import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, \
    ElementNotInteractableException


from Net.NNet import NNetWrapper
from training import OthelloMCTS
from Othello import OthelloGame, OthelloPlayer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LOGIN_PAGE = 'https://en.boardgamearena.com/account'

PLAYER_USER = 'USERNAME'
PLAYER_PASSWORD = 'PASSWORD'

DESIRED_PLAYER = None

WEIGHTS_FILE = 'othelo_model_weights'

degree_exploration = 1

num_simulations = 25

neural_network = NNetWrapper(board_size=(8,8))
if WEIGHTS_FILE:
    neural_network.load_checkpoint(WEIGHTS_FILE)

neural_networks_mcts = OthelloMCTS(8, neural_network, degree_exploration)


def click_on_region(driver, element):
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element_with_offset(element, 5, 5)
    action.click().perform()

driver = webdriver.Chrome()

driver.get(LOGIN_PAGE)

driver.implicitly_wait(5)

driver.find_element_by_id('username_input').send_keys(PLAYER_USER)
driver.find_element_by_id('password_input').send_keys(PLAYER_PASSWORD)
driver.find_element_by_id('submit_login_button').click()

time.sleep(5)

driver.refresh()

driver.find_element_by_xpath('//a[contains(text(), "Play now")]').click()

driver.find_element_by_xpath('//a[text()="Got it!"]').click()

driver.find_element_by_id('mobile_switcher_firstline_gamemode').click()
driver.find_element_by_id('pageheader_simple').click()
driver.find_element_by_id('mobile_switcher_secondline').click()
driver.find_element_by_id('pageheader_realtime').click()
driver.find_element_by_id('mobile_switcher_secondline_lobbymode').click()
driver.find_element_by_id('pageheader_manual').click()

driver.find_element_by_id('Xutton_play_35').click()

# driver.find_element_by_id('filter_name').send_keys('Reversi')
# driver.find_element_by_xpath('//span[contains(text(), "Play now!")]').click()
# driver.find_element_by_id('newArchiveCommentNext').click()
while True:
    try:
        driver.find_element_by_id('joingame_create_35').click()
        break
    except ElementClickInterceptedException:
        pass

if DESIRED_PLAYER:
    driver.find_element_by_id('inviteplayer').send_keys(DESIRED_PLAYER)
    driver.find_element_by_id('inviteplayer').click()
    time.sleep(2)
    driver.find_element_by_id('inviteplayer').send_keys(Keys.RETURN)

    while True:
        try:
            driver.find_element_by_id('startgame').click()
            break
        except NoSuchElementException:
            pass
        except ElementNotInteractableException:
            pass
else:
    driver.find_element_by_xpath('//span[text()="Open table to other players"]').click()
    while True:
        try:
            driver.find_element_by_id('ags_start_game_accept').click()
            break
        except NoSuchElementException:
            pass
        

opponent_user = None
opponent_desired = None

while not opponent_user:
    try:
        driver.implicitly_wait(1)
        opponent_user = set((el.text for el in driver.find_elements_by_class_name('playername') if el.text and el.text != PLAYER_USER))
        opponent_user = list(opponent_user)[0]
        break
    except Exception:
        pass

# while not opponent_desired:
#     opponent_user = None
#     while not opponent_user:
#         try:
#             driver.implicitly_wait(1)
#             opponent_user = set((el.text for el in driver.find_elements_by_class_name('playername') if el.text and el.text != PLAYER_USER))
#             opponent_user = list(opponent_user)[0]
#         except Exception:
#             pass
#     driver.implicitly_wait(5)
#     if DESIRED_PLAYER and opponent_user != DESIRED_PLAYER:
#         driver.find_element_by_id('ags_start_game_refuse').click()
#         driver.find_element_by_xpath('//a[text()="[Expel player]"]').click()
#     else:
#         opponent_desired = True

driver.implicitly_wait(5)


def get_board_state(driver):
    board = np.zeros((8, 8, 2), dtype=bool)
    discs_root = driver.find_element_by_id('discs')
    discs = {}
    for disc_el in discs_root.find_elements_by_class_name('disc'):
        player = OthelloPlayer.WHITE if 'disccolor_ffffff' in disc_el.get_attribute('class') else OthelloPlayer.BLACK
        position = disc_el.get_attribute('id').split('_')[1]
        position = int(position[1]) - 1, int(position[0]) - 1
        channel = 0 if player is OthelloPlayer.BLACK else 1
        board[position[0], position[1], channel] = 1
    return board

machine_piece = None

while True:
    its_you = None
    end_game = None
    try:
        driver.implicitly_wait(1)
        its_you = 'You' in driver.find_element_by_id('pagemaintitletext').text
        end_game = 'End of game' in driver.find_element_by_id('pagemaintitletext').text
    except NoSuchElementException:
        continue
    else:
        if not its_you:
            continue
        elif end_game:
            break

    driver.implicitly_wait(5)

    if not machine_piece:
        machine_piece = OthelloPlayer.BLACK if driver.find_element_by_id('move_nbr').text == '1' else OthelloPlayer.WHITE
    
    state = get_board_state(driver)

    print('MCTS Simluations')
    
    for _ in range(num_simulations):
        neural_networks_mcts.simulate(state, machine_piece)
    
    print('MCTS Simluations has finished')
    
    action_probabilities = neural_networks_mcts.get_policy_action_probabilities(state, 0)
    valid_actions = OthelloGame.get_player_valid_actions(state, machine_piece)
    best_action = max(valid_actions, key=lambda position: action_probabilities[tuple(position)])

    print(f'Action: {best_action}')

    driver.find_element_by_id(f'square_{best_action[1] + 1}_{best_action[0] + 1}').click()

    time.sleep(5)

print('The game has ended!')
