# https://www.youtube.com/watch?v=9QhkOVJXkvE
#Problem of a horizontal bar subjected to a force
#with an uncertain capacity

import pystra as ra
import numpy as np

limit_state = ra.LimitState(lambda R,S: R - S)

stochastic_model = ra.StochasticModel()

stochastic_model.addVariable(ra.Lognormal("R", 16.6, 1.7))

stochastic_model.addVariable( ra.Normal('S',10, 2))

options = ra.AnalysisOptions()
options.setPrintOutput(False)

Analysis = ra.Form(analysis_options=options,stochastic_model=stochastic_model, limit_state=limit_state)
Analysis.run() # run analysis

Analysis.showDetailedOutput()