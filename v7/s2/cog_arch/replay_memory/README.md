

store the most surprisive memories/states.....

THIS IS, JUST STORING THE OUTCOMES (STATE t + ACTION --> STATE t+1) WICH DIDNT YIELD WHERE EXPECTED


:::::::::::::::::::::::::::


La principal función de este módulo de Memoria es la de Reentrenar al "Cost module"

La ENTROPÍA vuelve a jugar un papel acá, puesto que guardaremos en Memoria del Agente Contexto aquellos States raros (los que presentan multiples niveles de Latent Variables)


:::::::::::::::::::::::::::::


Luego, la Memoria tendrá acceso al Agente para Reentrenar el Cost module por medio de "Experience Replay" mechanism

(Esto pasará durante el "sueño" del Agente, es decir, en cualquier momento mientras esté Inactivo)



