import unittest
import numpy as np
from OthelloGame import *


class TestStringMethods(unittest.TestCase):
    def test_base_functions_8x8(self):
        game = OthelloGame(8)
        self.assertEqual(game.n,8)
        start_board = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0,-1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1,-1, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))
        initBoard = game.getInitBoard()
        self.assertTrue((start_board==initBoard).all())
        self.assertEqual((8,8),game.getBoardSize())
        self.assertEqual(65,game.getActionSize())

        self.assertEqual(game.action_to_move(4),("a",4))
        self.assertEqual(game.action_to_move(27),("d",3))
        self.assertEqual(game.action_to_move(64),"skip")


        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 1, 1, 0, 0, 0],
                                    [ 0, 0, 0, 1,-1, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))

        after_board,after_player = game.getNextState(start_board,1,19)
        self.assertTrue((after_board==expected_board).all())
        self.assertEqual(after_player,-1)


        start_board = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 0, 0, 0, 0],
                                [ 0, 0,-1, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1,-1, 1, 0, 0],
                                [ 0, 0, 0,-1, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))

        expected_board =  np.array(([ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0,-1,-1, 1, 0, 0, 0],
                                    [ 0, 0, 0,-1,-1, 1, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))
        after_board,after_player =game.getNextState(start_board,-1,3)
        self.assertTrue((after_board==expected_board).all())
        self.assertEqual(after_player,1)

        start_board = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 0, 0, 0, 0],
                                [ 0, 0,-1, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1,-1, 1, 0, 0],
                                [ 0, 0, 0,-1, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))

        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 1,-1,-1, 0, 0, 0],
                                    [ 0, 0, 0,-1, 1,-1, 0, 0],
                                    [ 0, 0, 0, 1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))
        after_board = game.getCanonicalForm(start_board,-1)
        after_board2 = game.getCanonicalForm(start_board,1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==start_board).all())

        start_board = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 1, 1, 0],
                                [ 0, 0, 0, 1,-1,-1,-1, 0],
                                [-1,-1,-1, 1, 1, 0, 0, 0],
                                [ 1, 1, 1, 1,-1, 1, 0, 0],
                                [ 0, 0, 0,-1, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))

        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0,-1,-1,-1,-1, 0],
                                    [ 0, 0, 0,-1, 1, 1, 1, 0],
                                    [ 1, 1, 1,-1,-1, 0, 0, 0],
                                    [-1,-1,-1,-1, 1,-1, 0, 0],
                                    [ 0, 0, 0, 1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))
        after_board = game.getCanonicalForm(start_board,-1)
        after_board2 = game.getCanonicalForm(start_board,1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==start_board).all())



    
    def test_base_functions_6x6(self):
        game = OthelloGame(6)
        self.assertEqual(game.n,6)
        start_board = np.array(([ 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0],
                                [ 0, 0,-1, 1, 0, 0],
                                [ 0, 0, 1,-1, 0, 0],
                                [ 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0]))
        initBoard = game.getInitBoard()
        self.assertTrue((start_board==initBoard).all())
        self.assertEqual((6,6),game.getBoardSize())
        self.assertEqual(37,game.getActionSize())

        self.assertEqual(game.action_to_move(4),("a",4))
        self.assertEqual(game.action_to_move(15),("c",3))
        self.assertEqual(game.action_to_move(36),"skip")


        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 1, 0, 0],
                                    [ 0, 0, 1,-1, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0]))

        after_board,after_player = game.getNextState(start_board,1,8)
        self.assertTrue((after_board==expected_board).all())
        self.assertEqual(after_player,-1)


        start_board =     np.array(([ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0,-1, 0, 0, 0],
                                    [ 0, 0,-1, 1, 0, 0],
                                    [ 0, 0,-1,-1, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0]))

        expected_board =  np.array(([ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 1, 0, 0],
                                    [ 0, 0, 1,-1, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0]))
        after_board,after_player =game.getNextState(start_board,1,2)
        self.assertTrue((after_board==expected_board).all())
        self.assertEqual(after_player,-1)

        start_board =     np.array(([ 0, 0,-1, 0, 0, 0],
                                    [ 0, 0,-1, 0, 0, 0],
                                    [ 0, 0,-1,-1, 0, 0],
                                    [ 0, 0,-1, 1, 0, 0],
                                    [ 0, 0,-1, 1, 0, 0],
                                    [ 0, 0, 0, 1, 0, 0]))

        expected_board =  np.array(([ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 1, 0, 0],
                                    [ 0, 0, 1,-1, 0, 0],
                                    [ 0, 0, 1,-1, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0]))

        after_board = game.getCanonicalForm(start_board,-1)
        after_board2 = game.getCanonicalForm(start_board,1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==start_board).all())

        start_board =     np.array(([ 0, 0,-1, 0, 0, 0],
                                    [ 0, 0,-1, 0, 0, 0],
                                    [ 0, 0,-1,-1, 0, 0],
                                    [ 0,-1,-1, 1, 0, 0],
                                    [ 0, 1,-1, 1, 1, 0],
                                    [ 0, 1, 1, 1, 1, 0]))

        expected_board =  np.array(([ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0, 1, 1, 0, 0],
                                    [ 0, 1, 1,-1, 0, 0],
                                    [ 0,-1, 1,-1,-1, 0],
                                    [ 0,-1,-1,-1,-1, 0]))
        after_board = game.getCanonicalForm(start_board,-1)
        after_board2 = game.getCanonicalForm(start_board,1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==start_board).all())


    def test_value_functions_8x8(self):
        game = OthelloGame(8)
        test_board_1 =np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1, 1,-1, 0, 0],
                                [ 0, 0, 0, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))

        test_board_2 =np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 1, 0, 0],
                                [ 0, 0, 0, 1, 1, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 1, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))
        
        
        v = -4/15
        self.assertEqual(game.get_mobility_value(test_board_1,1,game.getValidMoves(test_board_1,1),game.getValidMoves(test_board_1,-1)), v)
        self.assertEqual(game.get_coin_value(test_board_1,1),5/7)
        self.assertEqual(game.get_coin_value(test_board_1,-1),-5/7)
        self.assertEqual(game.get_coin_value(test_board_2,1))

        

        


        
    
if __name__ == '__main__':
    unittest.main()