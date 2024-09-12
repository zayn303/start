import pandas as pd
from transformers import TapexTokenizer, BartForConditionalGeneration


def generate_response_from_table(table, query):
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    encoding = tokenizer(table=table, query=query, return_tensors="pt")
    outputs = model.generate(**encoding)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded_outputs


data = {
  "table": [
            {
            "r030":36,"txt":"Австралійський долар","rate":28.0834,"cc":"AUD","exchangedate":"30.08.2024"
            }
            ,{
            "r030":124,"txt":"Канадський долар","rate":30.6041,"cc":"CAD","exchangedate":"30.08.2024"
            }
            ,{
            "r030":156,"txt":"Юань Женьміньбі","rate":5.8067,"cc":"CNY","exchangedate":"30.08.2024"
            }
            ,{
            "r030":203,"txt":"Чеська крона","rate":1.8253,"cc":"CZK","exchangedate":"30.08.2024"
            }
            ,{
            "r030":208,"txt":"Данська крона","rate":6.1276,"cc":"DKK","exchangedate":"30.08.2024"
            }
            ,{
            "r030":344,"txt":"Гонконгівський долар","rate":5.2848,"cc":"HKD","exchangedate":"30.08.2024"
            }
            ,{
            "r030":348,"txt":"Форинт","rate":0.116389,"cc":"HUF","exchangedate":"30.08.2024"
            }
            ,{
            "r030":356,"txt":"Індійська рупія","rate":0.4911,"cc":"INR","exchangedate":"30.08.2024"
            }
            ,{
            "r030":360,"txt":"Рупія","rate":0.002678,"cc":"IDR","exchangedate":"30.08.2024"
            }
            ,{
            "r030":578,"txt":"Норвезька крона","rate":3.9282,"cc":"NOK","exchangedate":"30.08.2024"
            }
            ,{
            "r030":643,"txt":"Російський рубль","rate":0.45051,"cc":"RUB","exchangedate":"30.08.2024"
            }
            ,{
            "r030":702,"txt":"Сінгапурський долар","rate":31.653,"cc":"SGD","exchangedate":"30.08.2024"
            }
            ,{
            "r030":710,"txt":"Ренд","rate":2.3286,"cc":"ZAR","exchangedate":"30.08.2024"
            }
            ,{
            "r030":752,"txt":"Шведська крона","rate":4.0325,"cc":"SEK","exchangedate":"30.08.2024"
            }
            ,{
            "r030":756,"txt":"Швейцарський франк","rate":48.8034,"cc":"CHF","exchangedate":"30.08.2024"
            }
            ,{
            "r030":818,"txt":"Єгипетський фунт","rate":0.8472,"cc":"EGP","exchangedate":"30.08.2024"
            }
            ,{
            "r030":826,"txt":"Фунт стерлінгів","rate":54.3009,"cc":"GBP","exchangedate":"30.08.2024"
            }
            ,{
            "r030":840,"txt":"Долар США","rate":41.1901,"cc":"USD","exchangedate":"30.08.2024"
            }


    ],
    "query": "GBP rate of exchange"



}



table = pd.DataFrame.from_dict(data['table'])
table = table.astype(str)
response = generate_response_from_table(table, data['query'])
print(response)