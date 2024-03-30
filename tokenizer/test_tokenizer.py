import unittest
import tokenizer
import os

class TestTokenizer(unittest.TestCase):
    
    test_prompt="This is a test prompt."

    @unittest.skip("Pass testing for no useful features")
    def test_tokenize(self):
        # get home path from environment variable
        home = os.environ['HOME']

        tok=tokenizer.Tokenizer(home+"/Downloads/workspace/llama/tokenizer.model")

        # tok should not be none
        self.assertIsNotNone(tok)
    
        encoded =  tok.encode(self.test_prompt, True, True)
        self.assertIsNotNone(encoded)

        decoded = tok.decode(encoded)

        self.assertEqual(decoded, self.test_prompt)
        
