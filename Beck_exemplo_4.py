#Beck, p.123. Exemplo 4:
#requires: pip install pystra
import pystra as ra
import numpy as np

limit_state = ra.LimitState(lambda X1,X2,X3: X1 * X2 - X3)

stochastic_model = ra.StochasticModel()

stochastic_model.addVariable(ra.Normal("X1", 40, 5)) #sigma_y
stochastic_model.addVariable(ra.Normal("X2", 50, 2.5)) #W
stochastic_model.addVariable(ra.Normal("X3", 1000, 200)) #M

options = ra.AnalysisOptions()
options.setPrintOutput(False)

Analysis = ra.Form(analysis_options=options,stochastic_model=stochastic_model, limit_state=limit_state)
Analysis.run() # run analysis
Analysis.showDetailedOutput()

sorm = ra.Sorm(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state, form=Analysis)
sorm.run()
sorm.showDetailedOutput()