import streamlit as st
from PIL import Image
import ICRS

def page1():
    st.title("Visualisation")

    st.header("Age distribution")
    image1 = Image.open('age_distribution.png')
    st.image(image1, caption='Age distribution, majority of the data lies between 20 to 70 age', use_column_width=True)

    st.header("Count of users Country wise")
    image1 = Image.open('Count of users Country wise.png')
    st.image(image1, caption='Most of the Users are from USA, We can use this in model building for selecting the best books in the country', use_column_width=True)

    st.header("Books and their Ratings (Top 20)")
    image1 = Image.open('Books and their Ratings (Top 20).png')
    st.image(image1,
             caption='Top 20 books ratings',
             use_column_width=True)

    st.header("Authors and Ratings")
    image1 = Image.open('Authors and Ratings (Top 20).png')
    st.image(image1,
             caption='Authors and Ratings (Top 20)',
             use_column_width=True)

    st.header("Year of Publication count (1950 - 2010)")
    image1 = Image.open('Year of Publication count (1950 - 2010).png')
    st.image(image1,
             caption='Year of Publication count (1950 - 2010)',
             use_column_width=True)

    st.header("Top 20 Authors by books")
    image1 = Image.open('top 20 authors by book.png')
    st.image(image1,
             caption='Top 20 Authors by books',
             use_column_width=True)

    st.header("Top 20 Publishers of books")
    image1 = Image.open('top 20 publishers of books.png')
    st.image(image1,
             caption='Top 20 Publishers of books',
             use_column_width=True)

    st.header("Top 20 years in which books were published")
    image1 = Image.open('Top 20 years in which books were published.png')
    st.image(image1,
             caption='Top 20 years in which books were published',
             use_column_width=True)

    st.header("Percentage Data")
    image1 = Image.open('newplot1.png')
    st.image(image1,
             caption='Statistical Data',
             use_column_width=True)

    st.header("Most Rated Books by Users")
    image1 = Image.open('Most Rated Books by Users.png')
    st.image(image1,
             caption='Most Rated Books by Users',
             use_column_width=True)

    st.header("Top 10 rated books by the users")
    image1 = Image.open('Top 10 rated books by the users.png')
    st.image(image1,
             caption='Top 10 rated books by the users',
             use_column_width=True)


def page2():
    st.title("Book Recommendation System")
    text=st.text_input('Enter movie name for recommendation')
    submit_button = st.button("Submit")
    if text !='' and submit_button:

        recommendation_df=ICRS.item_based_coll_rs(text)
        if recommendation_df is not None:
            st.header("Our models Recommendation:")
            for (candidate_book, corr) in zip(recommendation_df['Book-Title'], recommendation_df['Correlation']):
                corr_thershold = 0.7
                if corr > corr_thershold:
                    ratings = ICRS.df[ICRS.df['Book-Title'] == candidate_book]['Book-Rating'].mean()
                    st.markdown('<span style="color: green;">{}</span>'.format(candidate_book), unsafe_allow_html=True)
                    st.markdown('<span style="color: yellow;">Ratings -></span>\n'.format(candidate_book), unsafe_allow_html=True)
                    st.markdown('<span style="color: red;">{}</span>'.format(str(round(ratings, 1))), unsafe_allow_html=True)
                    st.markdown("-" * 50)
                else:
                    break
        else:
            st.header("This book is not in our library, check out our most popular books:")

            ICRS.popular_books()


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Visualisation", "Book Recommendation System"))

    if page == "Visualisation":
        page1()
    elif page == "Recommendation System":
        page2()

if __name__ == "__main__":
    main()

# Harry Potter and the Prisoner of Azkaban (Book 3)
