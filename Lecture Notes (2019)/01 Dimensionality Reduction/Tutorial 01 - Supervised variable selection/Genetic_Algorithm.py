import numpy as np
import math

class Genetic_algorithm:

    def __init__(self,model,X,Y,chrom_num,eval_metric,chrom_ratio=0.5,top_num=2):
        '''
        :param model: models to fit
        :param X: input data
        :param Y: target data
        :param chrom_num: number of chroms (should be even number)
        :param eval_metric: metrics for evaluation, 'AIC', 'BIC', 'adj_Rsq' should be used
        :param chrom_ratio: probability of chroms to be 1
        :param top_num: number of parents that will transfer to next generation
        '''
        self.model = model
        self.X = X
        self.Y = Y
        self.chrom_num = chrom_num
        self.eval_metric = eval_metric
        self.chrom_ratio = chrom_ratio
        self.top_num = top_num

        self.variable_num = np.shape(X)[1]

    # metric function
    def metric(self,X,Y):
        model_fitted = self.model.fit(X, Y)
        yhat = model_fitted.predict(X)
        SSR = np.sum((np.mean(Y) - yhat) ** 2)
        SSE = np.sum((Y - yhat) ** 2)
        SST = SSR + SSE

        AIC = X.shape[0] + X.shape[0] * np.log(2*math.pi) + \
              X.shape[0] * np.log(SSE / X.shape[0]) + 2 * (X.shape[1] + 1)
        BIC = X.shape[0] + X.shape[0] * np.log(2*math.pi) + \
              X.shape[0] * np.log(SSE / X.shape[0]) + np.log(X.shape[0]) * (X.shape[1] + 1)
        r_sq = SSR/SST
        adj_r_sq = 1 - (1 - r_sq) * ( len(Y) - 1) / ( len(Y) - X.shape[1] - 1)

        return {"adj_Rsq": adj_r_sq, "AIC": AIC, "BIC":BIC}

    # finding top k index function
    def top_k_idx(self,input):
        return np.argpartition(input, -self.top_num)[-self.top_num:]

    # fitness evaluation function
    def fitness_eval(self,candidates):
        eval_values = []
        for i in range(len(candidates)):
            # error message when 0 variable selected
            if sum(candidates[i]) == 0:
                raise ValueError('0 variables selected, Please use chrom ratio greater than %.2f'%(self.chrom_ratio))
            selected_col = [c for c, v in zip(range(self.variable_num), candidates[i]) if v == 1]
            tmp_input = np.take(self.X, selected_col, axis=1)
            tmp_metric = self.metric(tmp_input, self.Y)[self.eval_metric]
            eval_values.append(tmp_metric)

        return eval_values

    # cross over function
    def cross_over(self, candidates, eval_values):

        probs = eval_values / sum(eval_values)

        # non-negative probability to 0
        for i in range(len(probs)):
            if probs[i] < 0:
                probs[i] = 0
            else:
                pass
        probs = probs / sum(probs)

        dart_num = int((self.chrom_num - 2) / 2)

        ### Cross over
        new_babies = []
        for i in range(dart_num):
            # select paranets
            selected_idx = np.random.choice(len(candidates), size=2, replace=False, p=probs)
            selected_babies = np.take(candidates, selected_idx, axis=0)

            # cross over
            cross_point = [s >= 0.5 for s in np.random.uniform(0, 1, size=self.variable_num)]

            new_baby_1 = []
            new_baby_2 = []

            # do cross over
            for i in range(len(cross_point)):
                if cross_point[i]:
                    new_baby_1.append(selected_babies[0][i])
                    new_baby_2.append(selected_babies[1][i])
                else:
                    new_baby_1.append(selected_babies[1][i])
                    new_baby_2.append(selected_babies[0][i])

            new_babies.append(new_baby_1)
            new_babies.append(new_baby_2)

        return new_babies

    # Mutate function
    def mutate(self,candidates,mutate_ratio):
        # generate mutents
        mutent = np.random.choice(a=[False, True], size=np.shape(candidates), p=[1 - mutate_ratio, mutate_ratio])
        # backup original set
        cand_bakup = np.copy(candidates)

        # change mutents
        if True in mutent:
            loc_f, loc_s = np.where(mutent)
            for i in range(len(loc_f)):
                if cand_bakup[loc_f[i]][loc_s[i]] == 1:
                    cand_bakup[loc_f[i]][loc_s[i]] = 0
                else:
                    cand_bakup[loc_f[i]][loc_s[i]] = 1
        else:
            pass

        return cand_bakup

    # Do Genetic Algorithm
    def Do_GA(self,max_iter,mutate_ratio=0.01):
        # Initialize
        babies = []
        for i in range(self.chrom_num):
            # generate one chrom
            baby = np.random.choice([0, 1], size=(self.variable_num,), p=[ 1 - self.chrom_ratio,self.chrom_ratio])
            # append
            babies.append(baby)

        # Iteration loop
        for i in range(max_iter):
            # Fitness evaluation
            eval_values = self.fitness_eval(candidates=babies)

            # Find top 2
            top_n = np.take(babies, self.top_k_idx(eval_values), axis=0)

            # Cross over
            new_babies = self.cross_over(candidates=babies, eval_values=eval_values)

            # Mutate
            mutent_babies = self.mutate(candidates=new_babies,mutate_ratio=mutate_ratio)

            # Merge top 2 and final variable set
            next_generation = np.vstack((top_n, mutent_babies))

            if (i + 1) % 10 == 0:
                print("Finished %dth generation !!" % (i + 1))

        # Final evaluation
        final_eval = self.fitness_eval(candidates=next_generation)
        final_select_idx = np.argmax(final_eval)
        final_variables = [c for c, v in zip(range(self.variable_num), babies[final_select_idx]) if v == 1]

        return final_variables, final_eval[final_select_idx]

