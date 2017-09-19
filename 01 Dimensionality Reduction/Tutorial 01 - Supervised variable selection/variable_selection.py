import numpy as np
from scipy.stats import t

class Variable_selection():

    def __init__(self,model,input_data,target_data):
        self.model = model
        self.input_data = input_data
        self.target_data = target_data

    def Sum_of_SQ(self,model, X, Y):
        yhat = model.predict(X)
        SSR = sum((np.mean(Y) - yhat) ** 2)
        SSE = sum((Y - yhat) ** 2)
        df_ssr = np.shape(X)[1]
        df_sse = np.shape(X)[0] - np.shape(X)[1]

        return SSR, SSE, df_ssr, df_sse

    def T_statistics(self,model, X, Y):
        params = np.append(model.intercept_, model.coef_)
        predictions = model.predict(X)

        newX = np.append(np.ones((len(X), 1)), X, axis=1)
        MSE = (sum((Y - predictions) ** 2)) / (len(newX) - len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

        return ts_b, p_values

    def F_statistics(self,model, X, Y):
        SSR, SSE, df_ssr, df_sse = self.Sum_of_SQ(model, X, Y)
        F = (SSR / df_ssr) / (SSE / df_sse)
        return F

    def R_sq(self,model, X, Y):
        model_trained = model.fit(X,Y)
        yhat = model_trained.predict(X)
        SSR = np.sum((np.mean(Y) - yhat) ** 2)
        SSE = np.sum((Y - yhat) ** 2)
        SST = SSR + SSE

        r_sq = 1 - (float(SSE)) / SST
        adj_r_sq = 1 - (1 - r_sq) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)

        return r_sq, adj_r_sq

    # selection cells
    def forward_cell(self, model, candidate_var, X, Y):
        initial_var = [n for n in range(0, np.shape(X)[1])]
        possible_list = np.delete(initial_var, candidate_var, 0).tolist()
        F_list = []

        # For all variables are selected
        if len(possible_list) == 0:
            tmpX = np.take(X, candidate_var, axis=1)
            model.fit(tmpX, Y)
            _, p_selected = self.T_statistics(model, tmpX, Y)
            return candidate_var, p_selected

        else:
            for i in range(len(possible_list)):
                tmp_variable = candidate_var + possible_list[i:i + 1]
                tmp_input = np.take(X, tmp_variable, axis=1)
                model.fit(tmp_input, Y)
                F = self.F_statistics(model, tmp_input, Y)

                F_list.append(F)

            selected_idx = np.argmax(F_list)
            result = candidate_var + [possible_list[selected_idx]]

            # for stopping
            model.fit(np.take(X, result, axis=1), Y)
            _, p_selected = self.T_statistics(model, np.take(X, result, axis=1), Y)

            return result, p_selected[1:]

    def backward_cell(self,model, candidate_var, X, Y):

        SSR_list = []
        for i in range(len(candidate_var)):
            tmp_var = candidate_var[:i] + candidate_var[i + 1:]
            tmp_input = np.take(X, tmp_var, axis=1)
            model.fit(tmp_input, Y)
            SSR, SSE, _, _ = self.Sum_of_SQ(model, tmp_input, Y)
            SSR_list.append(SSR)

        selected_idx = np.argmax(SSR_list)
        result = candidate_var[:selected_idx] + candidate_var[selected_idx + 1:]

        # for stopping
        model.fit(np.take(X, result, axis=1), Y)
        _, p_selected = self.T_statistics(model, np.take(X, result, axis=1), Y)

        return result, p_selected[1:]

    # forward selection
    def forward_selection(self,alpha):
        selected_var = []

        for i in range(np.shape(self.input_data)[1]):
            # calculate selected variable
            selected_var, p = self.forward_cell(self.model, selected_var, self.input_data, self.target_data)

            ## Stopping criteria
            # find values to remove
            zombies = [s >= alpha for s in p]

            if True in zombies:
                selected_var = [s for s, z in zip(selected_var, zombies) if not z]
                break

        return selected_var

    #### backward elimination ####
    def backward_elimination(self,alpha):

        selected_var = [n for n in range(0, np.shape(self.input_data)[1])]

        for i in range(np.shape(self.input_data)[1]):
            # calculate selected variable
            selected_var, p = self.backward_cell(self.model, selected_var, self.input_data, self.target_data)

            ## Stopping criteria
            # find values to remove
            zombies = [s >= alpha for s in p]

            if True not in zombies:
                break

        return selected_var

    #### stepwise selection ####
    def stepwise_selection(self,alpha):
        selected_var = []
        i = 0

        while len(selected_var) <= np.shape(self.input_data)[1]:

            if i <= 1:
                # do forward selection
                selected_var, p = self.forward_cell(self.model, selected_var, self.input_data, self.target_data)
            else:
                # backup for comparision
                var_before = np.copy(selected_var).tolist()
                # do forward selection
                selected_var, p = self.forward_cell(self.model, selected_var, self.input_data, self.target_data)
                # do forward selection
                selected_var, p = self.backward_cell(self.model, selected_var, self.input_data, self.target_data)

                ## Stopping criteria
                zombies = [s >= alpha for s in p]
                if var_before == selected_var:

                    if True in zombies:
                        selected_var = [s for s, z in zip(selected_var, zombies) if not z]
                        break
                    # prevent infinite roop
                    else:
                        selected_var, p = self.forward_cell(self.model, selected_var, self.input_data, self.target_data)
            i += 1

        return selected_var


