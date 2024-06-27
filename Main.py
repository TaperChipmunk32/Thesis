from DataExtraction import extract_data
from BERTModel import BERT_EXPERIMENT
from GPTModel import GPT_EXPERIMENT
from LSTMModel import LSTM_EXPERIMENT
import pandas as pd

def main():
    data = extract_data()
    books = ['Matthew', 'Mark', 'Luke', 'John']
    verses = []
    labels = []
    for book in books:
        for chapter in data['ESV'][book]:
            for verse in data['ESV'][book][chapter]:
                verses.append(data['ESV'][book][chapter][verse])
                labels.append(books.index(book))

    df = pd.DataFrame({'verse': verses, 'label': labels})
    
    BERT_EXPERIMENT(df)
    GPT_EXPERIMENT(df)
    LSTM_EXPERIMENT(df)


if __name__ == "__main__":
    main()