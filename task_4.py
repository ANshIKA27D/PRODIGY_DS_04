import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\Users\ANJALI DUBEY\Downloads\archive\twitter_training.csv")
print(data.head())
col_names = ['Id', 'Entity', 'Emotion', 'Detail']
df = pd.read_csv(r"C:\Users\ANJALI DUBEY\Downloads\archive\twitter_training.csv", names=col_names)

print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
#....................................................................................
print(" ")
print("Frequency Of each emotion")
Emotion_counts = df['Emotion'].value_counts()
print(Emotion_counts)
#PLot for showing Different Emotion Frequency
plt.figure(figsize=(6, 6))
palette = sns.color_palette("Greens", n_colors=len(Emotion_counts))

wedges, texts, autotexts = plt.pie(
    Emotion_counts,
    labels=Emotion_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=palette,
    wedgeprops=dict(width=0.4) 
)

plt.title('Sentiment Distribution')
plt.show()


#---------------------------------------------------------------------------------------
brand_data = df[df['Entity'].str.contains('Facebook', case=False)]
brand_sentiment_counts = brand_data['Emotion'].value_counts()
print(brand_sentiment_counts)


#plot for different Frequncy of emotion with different brand
selected_brands = ['Google', 'Facebook', 'Microsoft', 'Nvidia']
filtered_df = df[df['Entity'].isin(selected_brands)]
brand_sentiment = filtered_df.groupby(['Entity', 'Emotion']).size().reset_index(name='Count')

plt.figure(figsize=(10, 6))
sns.barplot(data=brand_sentiment, x='Entity', y='Count', hue='Emotion', palette='Paired')
plt.title('Sentiment Distribution for Selected Brands')
plt.xlabel('Brand')
plt.ylabel('Number of Mentions')
plt.legend(title='Emotion')
plt.tight_layout()
plt.show() 
