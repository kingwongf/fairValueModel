import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ParticleFilter(object):
    def __init__(self, y, n_particle, sigma_2, alpha_2):
        self.y = y
        self.n_particle = n_particle
        self.sigma_2 = sigma_2
        self.alpha_2 = alpha_2
        self.log_likelihood = -np.inf
    
    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def F_inv(self, w_cumsum, idx, u):


        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        if u>=0.99:
            print(f"u: {u}")
            print(f"k: {k}")
        return k +1
        
    def resampling(self, weights):
        w_cumsum = np.cumsum(weights)
        idx = np.asanyarray(range(self.n_particle))
        k_list = np.zeros(self.n_particle, dtype=np.int32) # サンプリングしたkのリスト格納場所
        
        # 一様分布から重みに応じてリサンプリングする添え字を取得
        for i, u in enumerate(np.random.uniform(0, 1, size=self.n_particle)):
            k = self.F_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list

    # def resampling2(self, weights):
    #     """
    #     計算量の少ない層化サンプリング
    #     """
    #     idx = np.asanyarray(range(self.n_particle))
    #     u0 = rd.uniform(0, 1/self.n_particle)
    #     u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
    #     w_cumsum = np.cumsum(weights)
    #     k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
    #     return k
    
    def gen_price_diff_dist(self,t):
        self.price_diff_dist = np.asarray(
            np.unique(np.diff(self.y,1)[t-self.roll_window:t], return_counts=True)).T
        
        
        self.price_diff_dist[:,1] =self.price_diff_dist[:,1]/ self.price_diff_dist[:,1].sum()
        self.price_diff_dist = pd.DataFrame(self.price_diff_dist, columns=['price_diff', 'likelihood'])
        
    def likelihood_price_dist_lookup(self,y, x):
        
        
        
        ## shift price diff dist by particle price (x) to get price dist of x
        p_dist = self.price_diff_dist.copy(deep=True)
        p_dist.index = p_dist['price_diff'] + x

        idx_likelihood = np.searchsorted(p_dist.index, y, side='left')

        try:
            
            return p_dist.iloc[idx_likelihood]['likelihood']
        except:
            return p_dist.iloc[idx_likelihood-1]['likelihood']
    def gen_v(self):
        return np.random.choice(self.price_diff_dist.price_diff, size=1, p= self.price_diff_dist.likelihood)[0]

    
    def simulate(self, roll_window, seed=100):
        self.roll_window = roll_window
        np.random.seed(seed)

        # 時系列データ数
        T = len(self.y)
        
        # 潜在変数
        x = np.zeros((T+1, self.n_particle))
        x_resampled = np.zeros((T+1, self.n_particle))
        
        # 潜在変数の初期値
        # initial_x = rd.normal(0, 1, size=self.n_particle)
        
        initial_x = self.y[roll_window-1]

        x[roll_window] = x_resampled[roll_window] = initial_x

        # 重み
        w        = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T) # 時刻毎の尤度
        

        for t in range(T):
            if t < self.roll_window:
                pass

                
            else:
                print("\r calculating... t={}".format(t), end="")
                
                self.gen_price_diff_dist(t)
                
                # print(f"v dist: {np.average(self.price_diff_dist['price_diff'],weights=self.price_diff_dist['likelihood']) }")
                
                for i in range(self.n_particle):


                    # 1階差分トレンドを適用

                    ## draw sample noise from rolling window
                    # v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
                    v = self.gen_v()
                    
                    
                    x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
                    w[t, i] = self.likelihood_price_dist_lookup(self.y[t], x[t+1, i])
                w_normed[t] = w[t]/np.sum(w[t]) # 規格化
                l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度

                print(f"w_normed: {w_normed.shape}")
                print(f"x: {x.shape}")
                print(f"x_resampled: {x_resampled.shape}")

                # Resampling
                k = self.resampling(w_normed[t]) # リサンプルで取得した粒子の添字
                # k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
                # print(f" any index 100?: {np.any(k==100)}")
                assert np.any(k!=100)
                x_resampled[t+1] = x[t+1, k]
                
                # print(f"t: {t} x_resampled[t+1] :{x_resampled[t+1].mean()} y: {self.y[t]} x diff: {x[t+1].mean() - self.y[t]} resampled diff: {x_resampled[t+1].mean() - self.y[t]}")

                # 全体の対数尤度
                self.log_likelihood = np.sum(l) - T*np.log(self.n_particle)

                self.x = x
                self.x_resampled = x_resampled
                self.w = w
                self.w_normed = w_normed
                self.l = l

    def get_filtered_value(self):
        """
        尤度の重みで加重平均した値でフィルタリングされ値を算出
        """
        return np.diag(np.dot(self.w_normed, self.x[1:].T))
        
    def draw_graph(self):
        # グラフ描画
        T = len(self.y)
        
        t_x =  np.arange(self.roll_window, T)
        plt.figure(figsize=(30,20))
        plt.plot(t_x, self.y[self.roll_window:], c="b", label="simple_mid")
        plt.plot(t_x, self.get_filtered_value()[self.roll_window:], c="g", label="pred")
        
        for t in t_x:
            plt.scatter(np.ones(self.n_particle)*t, self.x[t], color="r", s=2, alpha=0.1)
        
        plt.legend()
        plt.title("log likelihood={:.3f}".format(self.log_likelihood))
        plt.show()