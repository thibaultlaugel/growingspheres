



class HCLS:
    """
    class to fit the Original Growing Spheres algorithm
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None,
                n_in_layer=2000, #en vrai n_in_layer depend de rayon pour garantir meme densite?
                first_radius=0.1,
                dicrease_radius=10,
                sparse=True,
                verbose=False):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        
        if target_class == None: #faux; en vrai la target class devrait target "une classe parmi les autres n'importe laquelle" quand c'est None
            target_class = 1 - np.argmax(self.y_obs)
        
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius 
        self.sparse = sparse
        
        self.verbose = verbose
        
        if float(self.y_obs[0]) != self.y_obs[0]:
            raise ValueError("Prediction function should return a probability vector (continuous values)!")
        
        
    def find_counterfactual(self):
        out = self.main_hcls()
    return out



    def main_hcls(self, B=1.0):
        max_iter = 100
        k = 0
        adv = self.obs_to_interprete.copy()
        class_obs = int(np.argmax(prediction_fn(self.obs_to_interprete.reshape(1,-1))))
        #target_class = int(1 - class_obs)
        while k < max_iter:
            adv = Local(adv, obs, B)
            if clf.predict_proba(adv.reshape(1, -1))[0][target_class] > 0.6:
                break
            k += 1
        if clf.predict_proba(adv.reshape(1, -1))[0][target_class] < 0.5:
            if B <= 200:
                #print('redoing HIll')
                Hill(obs, B= 2*B)
            else:
                raise ValueError('trop long Hill=====')  
        return adv


    def fonction_budget(z):
        return np.linalg.norm(z, ord=2)
    #en fonction d'un deplacement, calculer la distance associee

    def hi(i, w, l): #w est en fait w_i ici
        li = X[:, i].min(axis=0)
        ui = X[:, i].max(axis=0)
        return max(min(w/(1 + 2 * l), ui), li)

    '''def h_cap(w, l):
        l_ = X.min(axis=0)
        u_ = X.max(axis=0)
        return max(min(w/(1 + 2l), u_), l_)
    '''

    def bisection_cost(l, w):
        return fonction_budget(np.array([hi(i, w[i], l) for i in range(w.size)]))


    def search_lambda_bissection(w, B):
        #ls = np.arange(0.0, 1000.0, 1.0)
        #costs = [fonction_budget(np.array([hi(i, w[i], l) for i in range(w.size)])) for l in ls]
        k = bisection_search(B, (0, 100), w)
        return k

    def bisection_search(B, interval, w):
        f = lambda t: bisection_cost(t, w) - B
        a0 = interval[0]
        b0 = interval[1]
        m0 = (a0 + b0)/2
        epsilon = 0.1
        while abs(a0 - b0) > epsilon:
            if f(a0) * f(m0) <0:
                b0 = m0
            elif f(b0) * f(m0) < 0:
                a0 = m0
            else:
                #print('ERREUR')
                break
            m0 = (a0 + b0)/2
        return m0


    def proj(w, B):
        # check si deplacement cappe reste dans le budget
        # si oui on laisse tel quel
        # si non on cherche a saturer la contrainte en 
        if fonction_budget(np.array([hi(i, w[i], 0) for i in range(w.size)])) <= B:
            l = 0
        else:
            l = search_lambda_bissection(w, B)
        z = np.array([hi(i, w[i], l) for i in range(w.size)])
        return z

    def Local(adv, obs, B=1.0):
        ### EN GROS: On explore m dimensions a chaque fois et on update avec la meilleure
        # on genere m chiffres uniformement --> indices
        # on genere m perturbations gaussiennes
        # on update adv = adv + perturbation
        # deplacement = adv - obs
        # on projette deplacement
        # on se retrouve avec un deplacement sur m dimensions projete
        # si on ameliore l'objectif avec l(un de) ces m deplacements, alors on choisit le deplacement qui ameliorer le mieux
        m = max(int(X.shape[1] / 3), 3)
        class_obs = clf.predict(obs.reshape(1,-1))
        target_class = int(1 - class_obs)
        #for j in range(m):

        q = np.random.randint(0, X.shape[1], m)
        sigmaq = X[:,q].std(axis=0)
        b = np.random.normal(0.0, 1.0, m)
        beq = np.zeros((m, X.shape[1]))
        beq[range(m), q] = b
        adverses = np.array([adv] * m)
        obses = np.array( [obs] * m)
        z = adverses + beq - obses # deplacement depuis obs originale: adverse actuel - obs origine + nouveau deplacement
        z = np.array([proj(z[j], B) for j in range(m)]) #grosso merdo ca va etre z cappe

        if max(clf.predict_proba(obses + z)[:, target_class]) > clf.predict_proba(adv.reshape(1,-1))[0][target_class]: #sil y a progres: ici, code comme si on voulait
            zj = np.argmax(clf.predict_proba(obses + z)[:, target_class])
            adv = obs + z[zj] # plutot obs + zj non ? papier dit adv + zj
        return adv

    

