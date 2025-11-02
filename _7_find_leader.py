
import pickle


with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
    results = pickle.load(f)


# gamma = results['popularity_records']
# gamma_sorted = sorted([(float(key), float(val)) for key,val in gamma.items()], key=lambda x:x[1], reverse=True)

gamma = results['gamma']
gamma_sorted = sorted([(float(key), float(val)) for key,val in enumerate(gamma)], key=lambda x:x[1], reverse=True)

print(gamma_sorted)