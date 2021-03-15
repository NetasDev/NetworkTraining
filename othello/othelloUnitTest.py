import unittest
from OthelloGame import *
import numpy as np
import numpy.testing as nptest

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

        self.assertEqual(game.action_to_move(4),(0,4))
        self.assertEqual(game.action_to_move(27),(3,3))
        self.assertEqual(game.action_to_move(64),(8,0))


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
        after_board2 = game.getCanonicalForm(start_board,-1)
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

        start_board = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 1, 1, 0],
                                [ 0, 0, 0, 1,-1,-1,-1, 0],
                                [-1,-1,-1, 1, 1, 0, 0, 0],
                                [ 1, 1, 1, 1,-1, 1, 0, 0],
                                [ 0, 0, 0,-1, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0]))

        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 1, 1, 1, 1, 0],
                                    [ 0, 0, 0, 1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 1, 1, 0, 0, 0],
                                    [ 1, 1, 1, 1, 0, 1, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))

        expected_board2 = np.array(([ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0,-1,-1,-1, 0],
                                    [-1,-1,-1, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0,-1, 0, 0, 0],
                                    [ 0, 0, 0,-1, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0]))

        after_board = game.get_only_pieces_of_player(start_board,1)
        after_board2 = game.get_only_pieces_of_player(start_board,-1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==expected_board2).all())

    
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

        self.assertEqual(game.action_to_move(4),(0,4))
        self.assertEqual(game.action_to_move(15),(2,3))
        self.assertEqual(game.action_to_move(37),(7,0))


        expected_board =  np.array(([ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 1, 0, 0, 0],
                                    [ 0, 0,-1, 1, 0, 0],
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
        after_board2 = game.getCanonicalForm(start_board,-1)
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

        expected_board2 =  np.array(([ 0, 0, 1, 0, 0, 0],
                                     [ 0, 0, 1, 0, 0, 0],
                                     [ 0, 0, 1, 1, 0, 0],
                                     [ 0, 1, 1,-1, 0, 0],
                                     [ 0,-1, 1,-1,-1, 0],
                                     [ 0,-1,-1,-1,-1, 0]))

        after_board = game.get_only_pieces_of_player(start_board,1)
        after_board2 = game.get_only_pieces_of_player(start_board,-1)
        self.assertTrue((after_board==expected_board).all())
        self.assertTrue((after_board2==expected_board2).all())



        


        
        
        


        


"""
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
"""
if __name__ == '__main__':
    unittest.main()