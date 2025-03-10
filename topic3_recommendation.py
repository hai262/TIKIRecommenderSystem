#cài đặt các thư viện cần thiết:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import  cosine_similarity
from gensim import similarities
import matplotlib.pyplot as plt
import pickle as pickle 
import pickle
import streamlit as st
# from surprise import Reader
# from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection.validation import cross_validate
# import warnings
# warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(page_title="Recommender System", layout="wide", page_icon=":mag:")
# Source Code
pd.options.display.float_format = '{:.2f}'.format
#1. Products
# products = pd.read_csv('Products_now.csv')
# #2. Reviews
# reviews = pd.read_csv('Reviews.csv')
# #--------------
# GUI
# https://docs.streamlit.io/library/api-reference/layout
st.markdown("<h1 style='text-align: center; color: red;'>Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: darkblue;'>TIKI E-Commerce</h2>", unsafe_allow_html=True)
# GUI
menu = ["Business Objective","Build Project","Conclusion"]
choice = st.sidebar.selectbox('Menu',menu)
if choice == "Business Objective":
    st.markdown("<h3 style='text-align: left; color: darkblue;'>Business Objective</h3>", unsafe_allow_html=True)
    st.write("""
    * Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.
    * Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
    * Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?
    """)
    st.write("""
    + Mục tiêu/vấn đề: Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng. => Xây dựng các mô hình đề xuất:
      - Content-based filtering
      * Collaborative filtering
    """)
    st.image("onlineretail.jpg")
    st.markdown("""<h4 style='text-align: left; color: darkblue;'>
    Lựa chọn đề xuất nhóm mặt hàng điện tử từ Tiki
    </h4>""", unsafe_allow_html=True)
    st.markdown("""<h6 style='text-align: left; color: black;'>
    1. Phân bổ số lượng mã sản phẩm theo thương hiệu
    </h6>""", unsafe_allow_html=True)
    col1, col2 = st.columns((4,1))
    with col1: 
      st.image("probybrand.png",width=450)
    with col2:
      st.markdown("""<w6 style='text-align: left; color: blue;'>
        Nhận xét: Thương hiệu Samsung có số lượng mã nhiều nhất, thương hiệu Panasonic đứng thứ 2 và gần bằng 1/2 số lượng mã của Samsung. Các thương hiệu khác gần như có số lượng tương đồng nhau
        </w6>""", unsafe_allow_html=True)  
    st.markdown("""<h6 style='text-align: left; color: black;'>
    2. Phân bổ giá sản phẩm theo thương hiệu
    </h6>""", unsafe_allow_html=True)
    col1, col2 = st.columns((4,1)) 
    with col1: 
      st.image("pribybrand.png",width=400)
    with col2:
      st.markdown("""<w6 style='text-align: left; color: blue;'>
        Nhận xét: Giá bán trung bình thương hiệu surface cao nhất, Bosch xếp thứ 2 gần như xấp xỉ với giá Surface. Các sản phẩm của thương hiệu khác có khoảng giá gần như tương đổng nhau.
        </w6>""", unsafe_allow_html=True)  
    st.markdown("""<h6 style='text-align: left; color: black;'>
    3. Top 20 sản phẩm được đánh giá nhiều nhất
    </h6>""", unsafe_allow_html=True)
    col1, col2 = st.columns((4,1)) 
    with col1: 
      st.image("top20pro.png")
    with col2:
      st.markdown("""<w6 style='text-align: left; color: blue;'>
        Nhận xét: Chuột không dây logitech là sản phẩm có số lượng đánh giá nhiều nhất. Đứng thứ 2 là sản phẩm tai nghe nhét tai JBJ C15 có số lượng đánh giá gần bằng 1/2 chuột không dây logitech. Các sản phẩm khác có số lượng đánh giá tương đồng nhau
        </w6>""", unsafe_allow_html=True)   
    st.markdown("""<h6 style='text-align: left; color: black;'>
    4. Top 20 khách hàng thực hiện đánh giá nhiều nhất
    </h6>""", unsafe_allow_html=True)
    col1, col2 = st.columns((4,1)) 
    with col1: 
      st.image("top20user.png")
    with col2:
      st.markdown("""<w6 style='text-align: left; color: blue;'>
        Nhận xét: Khách hàng có id 7737978 có số lượng sản phẩm nhiều nhất 50 sản phẩm. Top 20 khách hàng review nhiều nhất đều có số lượng review từ 30 sản phẩm trở lên.
        </w6>""", unsafe_allow_html=True)   
elif choice=="Build Project":
    #1. Products
    # products = pd.read_csv('Products_now.csv')
    df1a = pd.read_csv('Products_now1.csv')
    df1b = pd.read_csv('Products_now2.csv')
    products = pd.concat([df1a,df1b],axis=0).reset_index(drop=True)
    st.markdown("<h3 style='text-align: left; color: darkblue;'>Build Project</h3>", unsafe_allow_html=True)
    menu1 = ["Content-based filtering","Collaborative filtering"]
    choice = st.sidebar.selectbox('Menu add',menu1)
    # Bước 4&5: Modeling & Evaluation/ Analyze & Report 
    # stopword
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'
    with open(STOP_WORD_FILE,'r',encoding='utf-8') as file:
      stop_words = file.read()
    stop_words = stop_words.split('\n')
    if choice == "Content-based filtering":
        col1, col2 = st.columns((1,1))
        with col1: 
          form = st.form(key='Select product')
          p_id = form.number_input(label='Enter product_id: ',step=1,)
          submit1 = form.form_submit_button(label='Submit1')
        with col2:
          form = st.form(key='Select recommend similar products')
          r_step = form.number_input(label='Enter recommended number of products: ',step=1)
          submit2 = form.form_submit_button(label='Submit2')          
        st.markdown("<h4 style='text-align: left; color: blue;'>Model 1. Cosine Similarity</h4>", unsafe_allow_html=True)
        # Giải pháp 1: Cosine_similarity
        #doc model count:
        pkl_tfidf = 'count_tfidf_matrix.pkl'
        with open(pkl_tfidf, 'rb') as file:
          tfidf_matrix = pickle.load(file)
        cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
        # với mỗi sản phẩm, lấy 10 sản phẩm tương quan nhất
        results = {}
        for idx, row in products.iterrows():    
            similar_indices = cosine_similarities[idx].argsort()[:-10:-1]
            similar_items = [(cosine_similarities[idx][i]) for i in similar_indices]
            similar_items = [(cosine_similarities[idx][i], products['item_id'][i]) for i in similar_indices]
            results[row['item_id']] = similar_items[1:]
        # Lấy thông tin sản phẩm
        def item(id):
          return products.loc[products['item_id']==id]['name'].to_list()[0].split('-')[0]
        # Thông tin sản phẩm gợi ý
        def recommend(item_id,num,products):
          st.markdown("""<w6 style='text-align: left; color: red;'>
          b. Recommending  %s  products similar to  %s ...
          </w6>""" %(str(num),item(item_id)), unsafe_allow_html=True)
          # st.write('b. Recommending '+str(num) + ' products similar to ('+ item(item_id)+ ')...')
          recs = results[item_id][:num]
          for i in range(0, int(round(num/2,0))):
            score1 = round(recs[i][0]*100,2)
            url1 = list(products[products['item_id']==recs[i][1]]['url'])[0]
            image1 = list(products[products['item_id']==recs[i][1]]['image'])[0]
            name1 = products.loc[products['item_id']==recs[i][1]]['name'].to_list()[0].split('-')[0]
            id1 = recs[i][1]
            score2 = round(recs[num-1-i][0]*100,2)
            url2 = list(products[products['item_id']==recs[num-1-i][1]]['url'])[0]
            image2 = list(products[products['item_id']==recs[num-1-i][1]]['image'])[0]
            name2 = products.loc[products['item_id']==recs[num-1-i][1]]['name'].to_list()[0].split('-')[0]
            id2 = recs[num-1-i][1]
            cols = st.columns((1,1,1,1))
            cols[0].markdown("Product ID: %s" % id1)
            cols[0].image(image1, width = 170)
            cols[1].markdown("- %s" % name1)
            cols[1].markdown("- Product score: %s" % score1)
            cols[1].markdown("- [Product link](%s)" % url1) 
            cols[2].markdown("Product ID: %s" % id2)
            cols[2].image(image2, width = 170)
            cols[3].markdown("- %s" % name2)
            cols[3].markdown("- Product score: %s" % score2)
            cols[3].markdown("- [Product link](%s)" % url2) 
        # WordCloud
        from wordcloud import WordCloud
        def get_product_text(item_id,num):
          rcmd_ids = [r[1] for r in results[item_id]]+ [item_id]
          text = (products[products['item_id'].isin(rcmd_ids)])
          return ' '.join(text.name + text.description)
        if submit1 or submit2:
          st.markdown("""<w6 style='text-align: left; color: red;'>
          a. Currently viewing products: 
          </w6>""", unsafe_allow_html=True)
          url = list(products[products['item_id']==p_id]['url'])[0]
          image = list(products[products['item_id']==p_id]['image'])[0]
          name = list(products[products['item_id']==p_id]['name'])[0]
          list_price = list(products[products['item_id']==p_id]['list_price'])[0]
          rating = list(products[products['item_id']==p_id]['rating'])[0]
          col1, col2 = st.columns((1,1))
          with col1:
            st.image(image, width = 220) 
          with col2:
            st.markdown("- Product name: %s" % name)
            st.markdown("- Product price: %s" % list_price)
            st.markdown("- Product rating: %s" % rating)
            st.markdown("- Check out this: [Product link](%s)" % url)        
          st.write(recommend(p_id,r_step,products))
          wordcloud_text = get_product_text(p_id,r_step)
          st.markdown("""<w6 style='text-align: left; color: red;'>
          c. Visualize Stopword:
          </w6>""", unsafe_allow_html=True)
          col1, col2 = st.columns((2,1))
          with col1: 
            fig, ax1 = plt.subplots()
            wc = WordCloud(stopwords=stop_words,max_words=120).generate(wordcloud_text)
            plt.axis('off')
            plt.imshow(wc)
            st.pyplot(fig)
          with col2:
            st.markdown("""<w6 style='text-align: left; color: blue;'>
        Nhận xét: Do sử dụng tên và mô tả để so sánh đặc điểm của sản phẩm, độ chính xác hệ thống sẽ bị ảnh hưởng khi tên và phần mô tả không chính xác hoặc thiếu thông tin, chưa thể hiện rõ các cụm từ chính liên quan đến sảm phẩm muốn đề xuất và nhiễu thông tin.
        </w6>""", unsafe_allow_html=True)

        # Giải pháp 2: Gensim
        st.markdown("<h4 style='text-align: left; color: blue;'>Model 2. Gensim</h4>", unsafe_allow_html=True)
        # Tính toán sự tương tự trong ma trận thưa thớt
        gen_tfidf = 'gensim_tfidf.pkl'
        with open(gen_tfidf, 'rb') as file:
          tfidf = pickle.load(file)    
        gen_corpus = 'gensim_corpus.pkl'
        with open(gen_corpus, 'rb') as file:
          corpus = pickle.load(file) 
        gen_dictionary = 'gensim_dictionary.pkl'
        with open(gen_dictionary, 'rb') as file:
          dictionary = pickle.load(file)      
        index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=19328)
        # When user choose one product 38458616
        product_ID = p_id
        product = products[products['item_id']==product_ID].head(1)
        # Sản phẩm đang xem 
        name_description_pre = product['name_description_pre'].to_string(index=False)
        # Đề xuất cho sản phẩm đang xem
        def recommender(view_product,dictionary,tfidf,index,p_id,num):
          # Convert search words into Sparse Vectors
          view_product = view_product.lower().split()
          kw_vector = dictionary.doc2bow(view_product)
          # similarity calculation
          sim = index[tfidf[kw_vector]]
          # print result
          list_id = []
          list_score = []
          for i in range(len(sim)):
            list_id.append(i)
            list_score.append(sim[i])
          df_result = pd.DataFrame({'id':list_id,
                                    'score':list_score})
          # Five (num) highest scores
          five_highest_score = df_result.sort_values(by='score',ascending=False).head(num+1)
          idToList = list(five_highest_score['id'])
          products_find = products[products['index'].isin(idToList)]
          results = products_find[['index','item_id','name']]
          results = pd.concat([results,five_highest_score],axis=1).sort_values(by='score',ascending=False)
          # return results 
          results = results[['score','item_id']].values.tolist()
          st.markdown("""<w6 style='text-align: left; color: red;'>
          b. Recommending  %s  products similar to  %s ...
          </w6>""" %(str(num),item(p_id)), unsafe_allow_html=True)
          # st.write('b. Recommending '+str(num) + ' products similar to ('+ item(item_id)+ ')...')
          recs = results[:num]
          for i in range(0, int(round(num/2,0))):
            score1 = round(recs[i][0]*100,2)
            url1 = list(products[products['item_id']==recs[i][1]]['url'])[0]
            image1 = list(products[products['item_id']==recs[i][1]]['image'])[0]
            name1 = products.loc[products['item_id']==recs[i][1]]['name'].to_list()[0].split('-')[0]
            id1 = int(recs[i][1])
            score2 = round(recs[num-1-i][0]*100,2)
            url2 = list(products[products['item_id']==recs[num-1-i][1]]['url'])[0]
            image2 = list(products[products['item_id']==recs[num-1-i][1]]['image'])[0]
            name2 = products.loc[products['item_id']==recs[num-1-i][1]]['name'].to_list()[0].split('-')[0]
            id2 = int(recs[num-1-i][1])
            cols = st.columns((1,1,1,1))
            cols[0].markdown("Product ID: %s" % id1)
            cols[0].image(image1, width = 170)
            cols[1].markdown("- %s" % name1)
            cols[1].markdown("- Product score: %s" % score1)
            cols[1].markdown("- [Product link](%s)" % url1) 
            cols[2].markdown("Product ID: %s" % id2)
            cols[2].image(image2, width = 170)
            cols[3].markdown("- %s" % name2)
            cols[3].markdown("- Product score: %s" % score2)
            cols[3].markdown("- [Product link](%s)" % url2) 
        if submit1 or submit2:
          st.markdown("""<w6 style='text-align: left; color: red;'>
          a. Currently viewing products: 
          </w6>""", unsafe_allow_html=True)
          url = list(products[products['item_id']==p_id]['url'])[0]
          image = list(products[products['item_id']==p_id]['image'])[0]
          name = list(products[products['item_id']==p_id]['name'])[0]
          list_price = list(products[products['item_id']==p_id]['list_price'])[0]
          rating = list(products[products['item_id']==p_id]['rating'])[0]
          col1, col2 = st.columns((1,1))
          with col1:
            st.image(image, width = 220) 
          with col2:
            st.markdown("- Product name: %s" % name)
            st.markdown("- Product price: %s" % list_price)
            st.markdown("- Product rating: %s" % rating)
            st.markdown("- Check out this: [Product link](%s)" % url) 
          st.write(recommender(name_description_pre,dictionary,tfidf,index,p_id,r_step))       
    elif choice == "Collaborative filtering":
        #2. Reviews
        # df = pd.read_csv('Reviews_temp.csv')
        # model_algorithm = 'svd_algorithm.pkl'
        # with open(model_algorithm, 'rb') as file:
        #   algorithm = pickle.load(file)
        # df['EstimateScore'] = df['item_id'].apply(lambda x: algorithm.predict(userId,x).est)
        # df = df.sort_values(by=['EstimateScore'],ascending=False)
        col1, col2 = st.columns((1,1))
        with col1: 
          form = st.form(key='Select product')
          p_id = form.number_input(label='Enter Customer_id: ',step=1)
          submit1 = form.form_submit_button(label='Submit1')
        with col2:
          form = st.form(key='Select recommend products')
          r_step = form.number_input(label='Enter recommended number of products: ',step=1)
          submit2 = form.form_submit_button(label='Submit2')          
        st.markdown("<h4 style='text-align: left; color: blue;'>Model 3. SVD</h4>", unsafe_allow_html=True)
        # Thông tin sản phẩm gợi ý
        def recommend(p_id,num,reviews):
          st.markdown("""<w6 style='text-align: left; color: red;'>
          b. Recommending  %s  products similar for Customer_Id  %s ...
          </w6>""" %(str(num),str(p_id)), unsafe_allow_html=True)
          reviews = reviews[reviews['customer_id']==p_id]
          rst = reviews[['EstimateScore','item_id']].values.tolist()
          recs = rst[:num]
          for i in range(0, int(round(num/2,0))):
            score1 = round(recs[i][0],2)
            url1 = list(reviews[reviews['item_id']==recs[i][1]]['url'])[0]
            image1 = list(reviews[reviews['item_id']==recs[i][1]]['image'])[0]
            name1 = reviews.loc[reviews['item_id']==recs[i][1]]['name'].to_list()[0].split('-')[0]
            id1 = int(recs[i][1])
            score2 = round(recs[num-1-i][0],2)
            url2 = list(reviews[reviews['item_id']==recs[num-1-i][1]]['url'])[0]
            image2 = list(reviews[reviews['item_id']==recs[num-1-i][1]]['image'])[0]
            name2 = reviews.loc[reviews['item_id']==recs[num-1-i][1]]['name'].to_list()[0].split('-')[0]
            id2 = int(recs[num-1-i][1])
            cols = st.columns((1,1,1,1))
            cols[0].markdown("Product ID: %s" % id1)
            cols[0].image(image1, width = 170)
            cols[1].markdown("- %s" % name1)
            cols[1].markdown("- Product rating: %s" % score1)
            cols[1].markdown("- [Product link](%s)" % url1) 
            cols[2].markdown("Product ID: %s" % id2)
            cols[2].image(image2, width = 170)
            cols[3].markdown("- %s" % name2)
            cols[3].markdown("- Product rating: %s" % score2)
            cols[3].markdown("- [Product link](%s)" % url2) 
        if submit1 or submit2:
          df = pd.read_csv('Reviews_temp.csv')
          # reader = Reader() 
          # data = Dataset.load_from_df(df,reader)
          # algorithm = SVD()
          # results = cross_validate(algorithm,data,measures=['RMSE','MAE'],cv=5,verbose=True)
          # # getting full dataset => fit model
          # trainset = data.build_full_trainset()
          # algorithm.fit(trainset)
          model_algorithm = 'svd_algorithm.pkl'
          with open(model_algorithm, 'rb') as file:
            algorithm = pickle.load(file)
          df['EstimateScore'] = df['item_id'].apply(lambda x: algorithm.predict(p_id,x).est)
          df = df.sort_values(by=['EstimateScore'],ascending=False)
          df = df.reset_index(drop=True)
          product_fil = products[['item_id','name','list_price','url','image']]
          reviews = pd.merge(df,product_fil,on='item_id').reset_index(drop=True)
          st.markdown("""<w6 style='text-align: left; color: red;'>
          a. Currently Customer ID: %s
          </w6>"""%p_id, unsafe_allow_html=True)
          revie = reviews[reviews['customer_id']==p_id]
          rev = revie[['customer_id','item_id','name','rating','EstimateScore','list_price']].sort_values('EstimateScore',ascending=False)[:r_step]
          st.dataframe(rev)
          st.write(recommend(p_id,r_step,reviews))
elif choice=="Conclusion":
    st.markdown("<h3 style='text-align: left; color: darkblue;'>Conclusion</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: blue;'>1. Model Content-based Filtering</h5>", unsafe_allow_html=True)
    st.write("""
      - Mô hình không cần bất kỳ dữ liệu nào về những người dùng khác, vì các đề xuất dành riêng cho người dùng này. Điều này giúp dễ dàng mở rộng quy mô đến một số lượng lớn người dùng.
      - Mô hình có thể nắm bắt sở thích cụ thể của người dùng và có thể đề xuất các mặt hàng thích hợp mà rất ít người dùng khác quan tâm.
      - Qua 2 thuật toán Cosine similarity và Gensim đều cho ra các đề xuất item liên quan cho người dùng dựa trên nội dung sử dụng các tính năng của mặt hàng để đề xuất các mặt hàng khác tương tự như những gì người dùng thích, dựa trên các hành động trước đây của họ hoặc phản hồi rõ ràng.
      - Hệ thống Đề xuất hoạt động dựa trên sự giống nhau giữa nội dung hoặc người dùng truy cập nội dung.
      - Giá trị đầu ra nằm trong khoảng từ 0-1. Trong thuật toán cosine cho ra hệ số đề xuất cao <0.5 tương đồng, còn thuật toán Gensim đề xuất các item có tổng score > 0.5 => thuật toán Gensim cho kết quả khả quan hơn thuật toán Cosine.
      - Tuy nhiên việc kết hợp giữa cột name và cột description để để so sánh đặc điểm của sản phẩm, độ chính xác hệ thống sẽ bị ảnh hưởng khi tên và phần mô tả không chính xác hoặc thiếu thông tin. Bên cạnh đó hệ số score đề xuất của 2 thuật toán với mỗi sản phẩm là khác nhau.
    """)
    st.markdown("<h5 style='text-align: left; color: blue;'>2. Model Collaborative Filtering</h5>", unsafe_allow_html=True)
    st.write("""
      - Mô hình sử dụng đồng thời các điểm tương đồng giữa người dùng và các sản phẩm để đưa ra các đề xuất, có thể đề xuất một sản phẩm cho người dùng này dựa trên sở thích của một người dùng tương tự khác
      - Mô hình có thể giúp người dùng khám phá những sở thích mới, có thể không biết người dùng quan tâm đến một mặt hàng nhất định, nhưng mô hình vẫn có thể đề xuất nó vì những người dùng tương tự cũng quan tâm đến mặt hàng đó
    """)
