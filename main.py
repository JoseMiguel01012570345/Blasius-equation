import data_generator
import engine
import boundary_conditions as bc

Data=data_generator.data(100)

X_train , X_val , Y_train , Y_val  = Data.storm_data()

print("data collected :)")
piin=engine.engine(X_train, X_val, Y_train, Y_val,epsilon=1e-300)