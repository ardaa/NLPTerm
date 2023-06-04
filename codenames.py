import random
import os
import termcolor
import gensim
import random
import time
import itertools
import tqdm
#gensim verbose
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

class Codenames:
    def __init__(self, model_name, row=5, col=5):
        self.import_model(model_name)
        #model min_count=5, epochs=10, window=5, sg=1, workers=4
        
        #self.model = gensim.models.KeyedVectors.load('codenames.model')
        #limit the model to 100000


        self.codenames_words = self.readfile()
        self.row = row
        self.col = col
        self.board = []
        self.blue_team = []
        self.red_team = []
        self.assassin = []
        self.neutral = []
        self.guessed_words = []
        self.blue_score = 0
        self.red_score = 0
        self.turn = "blue"
        self.winner = None
        self.lemmatizer = WordNetLemmatizer()

    def readfile(self):
        with open("./data/codenames_words.txt", "r") as f:
            codenames_words = f.read().splitlines()
        return codenames_words
    
    def lematize_word(self, word):
        return self.lemmatizer.lemmatize(word)

    def generate_board(self):
        words = random.sample(self.codenames_words, self.row * self.col)
        self.word_list = words
        self.blue_team = random.sample(words, 9)
        self.red_team = random.sample([word for word in words if word not in self.blue_team], 8)
        self.assassin = random.sample([word for word in words if word not in self.blue_team and word not in self.red_team], 1)
        self.neutral = [word for word in words if word not in self.blue_team and word not in self.red_team and word not in self.assassin]
        
        self.board = [words[i:i+self.col] for i in range(0, len(words), self.col)]
    
    def create_dataset(self):
        #create board, ask for hint, ask for number of words, ask for words, save to file, loop
        self.generate_board()
        self.print_board_colored()
        with open("./data/codenames_dataset.txt", "a") as f:
            f.write("________________________\n")
            #write board to file
            for row in self.board:
                f.write(f"{row}\n")
            f.write("________________________\n")

        while self.winner == None:
            hint = input("Enter a hint: ")
            num = int(input("Enter a number: "))
            words = []
            for i in range(num):
                word = input("Enter a word: ")
                #check if word is on the board, ask again if not
                while word.lower() not in [word.lower() for row in self.board for word in row]:
                    print("Word not on board")
                    word = input("Enter a word: ")
                words.append(word)
            with open("./data/codenames_dataset.txt", "a") as f:
                f.write(f"{hint} {num} {words}\n")
            self.turn = "blue" if self.turn == "red" else "red"
            self.print_board_colored()
            self.check_winner()
        print(f"{self.winner} team wins!")


    def tune_model(self):
        print("Tuning model...")
        start = time.time()



        #tune the googlenews word vectors to include the codenames words in the vocabulary
        tuned_word2vec = gensim.models.Word2Vec(vector_size=300, min_count=1, epochs=20)
        print(tuned_word2vec.epochs)

        normalized_words = [word.lower() for word in self.codenames_words]      
        #remove spaces from words
        normalized_words = [word.replace(" ", "_") for word in normalized_words]  
        tuned_word2vec.build_vocab(normalized_words)
        tuned_word2vec.build_vocab([list(self.model.index_to_key)], update=True)
        # Transfer intersecting word vectors from the loaded Google News model to the tuned_word2vec model
        weights = tuned_word2vec.wv
        print(random.sample(self.model.index_to_key, 10))
        intersecting_words = [word for word in tuned_word2vec.wv.key_to_index if word in self.model]
        weights.add_vectors(intersecting_words, self.model[intersecting_words])
        print(tuned_word2vec.corpus_count)
        # Continue training the tuned_word2vec model on your specific data
        tuned_word2vec.train(normalized_words, total_examples=tuned_word2vec.corpus_count, epochs=tuned_word2vec.epochs)
        tuned_word2vec.save('codenames.model')
        self.model = tuned_word2vec
        end = time.time()
        print(f"Model tuned in {end - start} seconds")

        #evaluate
        print(self.model.wv.most_similar("dog"))
        print(self.model.wv.most_similar("cat"))

        print(self.model.wv.similarity("dog", "cat"))
        print(self.model.wv.similarity("dog", "car"))




    def import_model(self, model):
        #import the tuned model
        if model == "bert":
            self.model_name = 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForNextSentencePrediction.from_pretrained(self.model_name)
        elif model == "word2vec":
            self.model_name = 'word2vec'
            self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def print_board_colored(self):
        #ascii seperator, align words, color words to their team color


        print("  " + "-" * (self.col * 12 + 1))
        for i in range(self.row):
            print("| ", end="")
            for j in range(self.col):
                if self.board[i][j] in self.blue_team:
                    print(termcolor.colored(self.board[i][j].ljust(10), "blue"), end="")
                elif self.board[i][j] in self.red_team:
                    print(termcolor.colored(self.board[i][j].ljust(10), "red"), end="")
                elif self.board[i][j] in self.assassin:
                    print(termcolor.colored(self.board[i][j].ljust(10), "magenta"), end="")
                else:
                    print(self.board[i][j].ljust(10), end="")
            print("|")
        print("  " + "-" * (self.col * 12 + 1))
        print(f"Blue Team: {self.blue_score} | Red Team: {self.red_score}")
        print(f"It is {self.turn}'s turn")
        
    def print_board(self):
        #ascii seperator, align words, color words to their team color


        print("  " + "-" * (self.col * 12 + 1))
        for i in range(self.row):
            print("| ", end="")
            for j in range(self.col):
                #if the word has been guessed, print it in the color of the team that guessed it
                if self.board[i][j] in self.guessed_words:
                    if self.board[i][j] in self.blue_team:
                        print(termcolor.colored(self.board[i][j].ljust(10), "blue"), end="")
                    elif self.board[i][j] in self.red_team:
                        print(termcolor.colored(self.board[i][j].ljust(10), "red"), end="")
                    elif self.board[i][j] in self.assassin:
                        print(termcolor.colored(self.board[i][j].ljust(10), "magenta"), end="")
                    elif self.board[i][j] in self.neutral:
                        print(termcolor.colored(self.board[i][j].ljust(10), "yellow"), end="")
                        
                #if the word has not been guessed, print it in white
                else:
                    print(self.board[i][j].ljust(10), end="")
            print("|")
        print("  " + "-" * (self.col * 12 + 1))
        print(f"Blue Team: {self.blue_score} | Red Team: {self.red_score}")
        print(f"It is {self.turn}'s turn")
    
    def check_winner(self):
        if self.blue_score == 9:
            self.winner = "blue"
        elif self.red_score == 8:
            self.winner = "red"
        return self.winner

    def get_hint(self):
        hint = input("Enter a hint: ")
        num = int(input("Enter a number: "))
        self.guess_word(hint, num)

    def guess_nlp(self, hint, num):
        #find the words that are closest to the hint on the board
        #print those words
        word_scores = {}
        #flatten board
        words = [word for row in self.board for word in row]
        words = [word.lower() for word in words]
        words = [word.replace(" ", "_") for word in words]  

        for word in words:
            #get word similarity
            try:
                word_scores[word] = self.model.similarity(word, hint)
            except KeyError:
                print(f"{word} not in vocabulary")
                continue
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        print("The answer are the following words:")
        for i in range(num):
            print(sorted_words[i][0])
            if sorted_words[i][0] in self.blue_team:
                self.blue_score += 1
            elif sorted_words[i][0] in self.red_team:
                self.red_score += 1
            elif sorted_words[i][0] in self.assassin:
                self.winner = self.turn
                break
        self.turn = "blue" if self.turn == "red" else "red"
        self.print_board_colored()
        self.check_winner()
        if self.winner == None:
            self.get_hint()
        else:
            print(f"{self.winner} team wins!")

    def guess_word(self, hint, num):
        for i in range(num):
            print(f"Hint: {hint, num}")
            print(f"Guess {i+1} of {num}")


            word = input("Enter a word: ")
            #check if word is on the board, ask again if not
            while word.lower() not in [word.lower() for row in self.board for word in row]:
                print("Word not on board")
                word = input("Enter a word: ")

            word = word.upper()
            self.guessed_words.append(word)
            if word in self.blue_team:
                self.blue_score += 1
            elif word in self.red_team:
                self.red_score += 1
            elif word in self.assassin:
                self.winner = "blue" if self.turn == "red" else "red"
            self.print_board()
        self.turn = "blue" if self.turn == "red" else "red"


    def find_similars(self, words):
        if self.model_name == 'word2vec':
            hints = self.model.most_similar(
            positive=words,
            topn=1,
            restrict_vocab=100000,
            )     
        elif self.model_name == 'bert-base-uncased':
            # Tokenize the input group of words
            tokenized = self.tokenizer.encode_plus(
                words,
                add_special_tokens=True,
                return_tensors='pt'
            )

            # Generate predictions using the pre-trained BERT model
            with torch.no_grad():
                outputs = self.model(**tokenized)
                predictions = outputs.logits

            # Get the index of the most related word
            related_word_index = torch.argmax(predictions, dim=1).item()

            # Decode the related word from its index
            hints = self.tokenizer.decode(related_word_index)
        return hints
    
    def find_hint(self):
        #try to find a hint that is similar to the words on the board, do it in pairs of 2, 3, 4 words, choose the one with the highest average similarity
        #create all possible combinations of 2, 3, 4 words
        wordlist = self.blue_team if self.turn == "blue" else self.red_team
        #drop solved words
        wordlist = [word for word in wordlist if word not in self.guessed_words]
        negativelist = self.red_team if self.turn == "blue" else self.blue_team
        negativelist.append(self.assassin[0])
        negativelist = [word.lower() for word in negativelist]
        negativelist = [word.replace(" ", "_") for word in negativelist]
        possible_words = []
        scores = {}

        for i in range(2, 5):
            possible_words.extend(list(itertools.combinations(wordlist, i)))

        for words in tqdm.tqdm(possible_words):
            break_outer = False
            words = [word for word in words]
            words = [word.lower() for word in words]
            words = [word.replace(" ", "_") for word in words]
            #drop words that are in the solved list
            #drop words that are in the negative list
            hints = self.find_similars(words)
            lematized_hint = self.lematize_word(hints[0][0]).lower()
            for word in words:
                score = self.model.similarity(word, hints[0][0])
                print(word,hints[0][0] ,score, words)
                if (lematized_hint in word) or (score < 0.1) or (word in lematized_hint):
                    break_outer = True
                    break
            if break_outer:
                continue
                
            scores[hints[0][0]] = [hints[0][1], len(words), words]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        print(len(sorted_scores))
        return sorted_scores[0][0], sorted_scores[0][1][1]
    
    def play_with_hint(self):
        hint, num = self.find_hint()
        print(f"Hint: {hint} {num}")
        self.guess_word(hint, num)

    def play(self):
        self.generate_board()
        self.print_board()
        while self.winner == None:
            if self.turn == "blue":
                self.play_with_hint()
            else:
                self.play_with_hint()
        print(f"{self.winner} team wins!")



if __name__ == "__main__":
    codenames = Codenames(model_name="bert")
    #codenames.tune_model()
    codenames.play()
        
        


    
