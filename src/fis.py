import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

class fis:

    def __init__(self):
        self.prop = np.arange(-1.02,1.01,0.0007)
        self.der = np.arange(-1.02,1.01,0.0007)
        self.sal_mot = np.arange(-1.07,1.07,0.0007)

        # Antecedente de la entrada proporcional P y sus valores para conjuntos triangulares

        self.proporcional = ctrl.Antecedent(self.prop, 'proporcional')
        self.proporcional['muy lejos'] = fuzz.trapmf(self.proporcional.universe, [-6, -4, -1, -0.4])
        self.proporcional['lejos'] = fuzz.trapmf(self.proporcional.universe, [-1, -0.4, -0.4, 0])
        self.proporcional['ok'] = fuzz.trapmf(self.proporcional.universe, [-0.4, 0, 0, 0.4])
        self.proporcional['cerca'] = fuzz.trapmf(self.proporcional.universe, [0, 0.4, 0.4, 1])
        self.proporcional['muy cerca'] = fuzz.trapmf(self.proporcional.universe, [0.4, 1, 4, 6])


        # Se define la variable de entrada Derivativo
        self.derivativo = ctrl.Antecedent(self.der, 'derivativo')
        self.derivativo['alejandose'] = fuzz.trapmf(self.derivativo.universe, [-6,-4,-0.2, 0])
        self.derivativo['sin cambio'] = fuzz.trapmf(self.derivativo.universe, [-0.2, 0, 0, 0.2])
        self.derivativo['acercandose'] = fuzz.trapmf(self.derivativo.universe, [0, 0.2, 4, 6])

        # Se define la variable de conjutns para los motores
        self.salida = ctrl.Antecedent(self.sal_mot, 'salida')
        self.salida['muy rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-1.02,-1.01,-0.90,-0.80])
        self.salida['bastante rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-0.90,-0.80,-0.80,-0.40])
        self.salida['más o menos rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-0.80,-0.40,-0.40,-0.30])
        self.salida['despacito hacia atrás'] = fuzz.trapmf(self.salida.universe, [-0.40,-0.30,-0.30,-0.10])
        self.salida['muy lento hacia atrás'] = fuzz.trapmf(self.salida.universe, [-0.30,-0.10,-0.10,0.10])
        self.salida['muy lento hacia delante'] = fuzz.trapmf(self.salida.universe, [-0.10,0.10,0.10,0.30])
        self.salida['despacito hacia delante'] = fuzz.trapmf(self.salida.universe, [0.10,0.30,0.30,0.50])
        self.salida['medio rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [0.30,0.50,0.50,0.60])
        self.salida['rápidamente hacia delante'] = fuzz.trapmf(self.salida.universe, [0.50,0.60,0.60,0.80])
        self.salida['bastante rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [0.60,0.80,0.80,0.90])
        self.salida['muy rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [0.80,0.90,1.01,1.02])


    def get_membership(self,antecedent_name, value, method:str = 'max'):
        # Verificar que el antecedente dado existe
        antecedent = None
        if antecedent_name == 'proporcional':
            antecedent = self.proporcional
        elif antecedent_name == 'derivativo':
            antecedent = self.derivativo
        elif antecedent_name == 'salida':
            antecedent = self.salida
        else:
            return "Antecedente no válido, pertenencia no calculada"

        # Calcular el valor de pertenencia de cada conjunto difuso para el valor dado
        memberships = {}
        for term in antecedent.terms:
            memberships[term] = fuzz.interp_membership(antecedent.universe, antecedent[term].mf, value)


        if method == 'max':
            max_value = max(memberships.values())
            key = [k for k, v in memberships.items() if v == max_value][0]
            return (key, max_value)
        else:
            membership_selected = {k: v for k, v in memberships.items() if v != 0.0}
            return membership_selected 
        



    
        
