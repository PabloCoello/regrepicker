import pandas as pd
import numpy as np
import math
from scipy.stats import f,t
from tabulate import tabulate



class ols:
        '''
        Methods for OLS estimation.
        '''
        def __init__(self, data, y, x):
                self.data_ = data
                self.y_ = y
                self.x_ = x
                self.Y = self.data_[self.y_]
                self.X = np.matrix(self.get_X_matrix())
                self.var_matrix = np.linalg.inv(np.transpose(self.X)*self.X)

                self.n = len(self.Y)
                self.k = len(self.x_)+1

                self.B = self.get_estimators()
                self.estimation = self.get_estimation()

                self.ssr = self.get_ssr()
                self.sse = self.get_sse()
                self.sst = self.get_sst()

                self.var = self.get_y_variance()
                self.residual_var = self.get_y_residual_variance()

                self.ssrh0 = self.get_ssrh0()
                self.fvalue = self.f_contrast()
                self.r2 = self.r2()

                self.sigmasq = self.get_sigmasq()
                self.sigma = math.sqrt(self.sigmasq)

                self.var_estimators = [(self.var_matrix[i,i]*self.sigmasq) for i in range(self.k)]
                self.se_estimators = [math.sqrt(var) for var in self.var_estimators]

                self.tvalues = self.t_contrast()
                self.formula = self.get_summary_formula()
                self.coefficients = self.get_summary_coefficients()
                self.statistics = self.get_summary_statistics()
                        
        
        def get_X_matrix(self):
                '''
                Returns regressor matrix.
                '''
                X = self.data_[x].assign(a=1)
                cols = X.columns.tolist()
                cols = cols[-1:]+cols[:-1]
                X = X[cols]
                return(X)


        def get_estimators(self):
                '''
                Returns OLS estimators array.
                '''
                B = (np.linalg.inv(np.transpose(self.X)*self.X) * 
                        np.transpose(self.X) * 
                        np.transpose(np.matrix(self.Y)))
                return(B)  


        def get_estimation(self):
                '''
                Returns the values estimated by the model of the explained variable.
                '''
                estimation = [float(np.transpose(self.B)*np.transpose(line)) for line in self.X]
                return(estimation)


        def get_sse(self):
                '''
                Returns the part of the variance explained by the model.
                '''
                sse = sum([(self.estimation[i]-(sum(self.Y)/len(self.Y)))**2 for i in range(len(self.Y))])
                return(sse)


        def get_ssr(self):
                '''
                Returns the part of the variance that can not be explained by the model.
                '''
                ssr = sum([(self.Y[i]-self.estimation[i])**2 for i in range(len(self.Y))])
                return(ssr)


        def get_sst(self):
                '''
                Returns total variation of dependent variable.
                '''
                sst = sum([(self.Y[i]-(sum(self.Y)/len(self.Y)))**2 for i in range(len(self.Y))])
                return(sst)


        def get_y_variance(self):
                '''
                Returns y variance.
                '''
                yvariance = self.sst/(self.n-1)
                return(yvariance)


        def get_y_residual_variance(self):
                '''
                Returns y residual variance.
                '''
                residual_yvariance = self.ssr/(self.n-self.k-1)
                return(residual_yvariance)


        def get_ssrh0(self):
                '''
                Returns ssr under h0:b1=b2=...=bn=0
                '''
                ssrh0 = float(sum([(self.Y[i]-self.B[0])**2 for i in range(len(self.Y))]))
                return(ssrh0)


        def get_sigmasq(self):
                sigmasq = self.ssr/(self.n-2)
                return(sigmasq)


        def f_contrast(self):
                '''
                Returns p-value for F_contrast: H0: b0 = b1 = b2 = ... = bn = 0
                '''
                fvalue = ((self.ssrh0-self.ssr)/(self.k-1))/(self.ssr/(self.n-self.k))
                pvalue = f.pdf(fvalue, self.k-1, self.n-self.k)
                return(pvalue)


        def t_contrast(self):
                '''
                Returns p-value for t_contrast of individual significance of regressors.
                '''
                array = []
                for i in range(len(self.se_estimators)):
                        tvalue = self.B[i]/self.se_estimators[i]
                        pvalue = t.pdf(tvalue, self.n-self.k)
                        array.append(float(pvalue))
                return(array)
                
                      
        def r2(self):
                '''
                Returns R squared of the regression.
                '''
                r2 = (self.sse/self.sst)
                return(r2)
        
        def get_summary_formula(self):
                '''
                Returns summary with the results
                '''
                # Formula
                regresor = ' + '.join(self.x_)
                formula = self.y_+' = intercept + '+ regresor
                return(formula)
        
        
        def get_summary_coefficients(self):
                # Coefficients table
                table = []
                names = self.x_
                names.append('intercept')
                for i in range(len(names)):
                        row = []
                        row.append(names[i-1])
                        row.append(float(self.B[i]))
                        row.append(self.se_estimators[i])
                        row.append(self.tvalues[i])
                        table.append(row)
                coefficients = tabulate(table,
                                        headers=['Coefficients', 
                                                 'Estimate', 
                                                 'Std. Error', 
                                                 'pvalue (t-test)'])
                return(coefficients)
        
                
        def get_summary_statistics(self):
                # Statistics
                stat_name = ['Std. residual','R-squared','F-statistic']
                statistics = tabulate([[stat_name[0], math.sqrt(self.residual_var)],
                                       [stat_name[1], self.r2],
                                       [stat_name[2], self.fvalue]],
                                      headers=['Statistic', 'Value'])
                return(statistics)
        
        def summary(self):
                return(print(self.formula +'\n'+'\n'+ self.coefficients +'\n'+'\n'+ self.statistics))

                



if __name__ == '__main__':
        data = pd.read_excel('../DATA/encuestas.xlsx')
        y = "PSOE"
        x = ["VOX_busc", "Ciudadanos_busc"]

        ls = ols(data, y, x)
        ls.summary()
