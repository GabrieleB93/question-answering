import pandas as pd
import json_lines

db_data = []
db_data1 = []
db_cols = ['url', 'date']


def main():
    with json_lines.open('data1.jsonl.gz') as f:
        for item in f:
            db_data.append(item)

    df = pd.DataFrame(db_data, columns=db_cols)
    print("GZ version")
    print(df)

    with open('data.jsonl', 'rb') as f:
        for item in json_lines.reader(f):
            db_data1.append(item)

    df1 = pd.DataFrame(db_data1, columns=db_cols)
    print("Json version")
    print(df1)


if __name__ == '__main__':
    main()
