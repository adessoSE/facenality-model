# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:08:30 2019

@author: grilborzer
"""

import os


def create_classification_folderstructure():
    traits = ["A-Warmth", "B-Reasoning", "C-Emotional-Stability", "E-Dominance", "F-Liveliness", "G-Rule-Consciousness", "H-Social-Boldness", "I-Sensitivity", "L-Vigilance", "M-Abstractedness", "N-Privateness", "O-Apprehension", "Q1-Openness-to-Change", "Q2-Self-Reliance", "Q3-Perfectionism", "Q4-Tension"]
    trait_high = "high-range"
    trait_low = "low-range"
    
    for trait in traits:
        if not os.path.exists(trait):
            os.makedirs(trait)
        
        if not os.path.exists(trait + "/" + trait_high):
            os.makedirs(trait + "/" + trait_high)
        
        if not os.path.exists(trait + "/" + trait_low):
            os.makedirs(trait + "/" + trait_low)
            

if __name__ == '__main__':
    create_classification_folderstructure()