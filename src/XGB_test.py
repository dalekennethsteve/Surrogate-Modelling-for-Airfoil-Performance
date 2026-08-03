import pandas as pd
import joblib

# Load
xgb_cl = joblib.load('xgb_cl.pkl')
xgb_cd = joblib.load('xgb_cd.pkl')

# E22
#cst_weights = [-0.14694, -0.1035, -0.11096, -0.07757, -0.07464, -0.08179,0.180194, 0.227316, 0.207657, 0.161912, 0.233228, 0.215928]

#G56
# cst_weights = [-0.1334,	-0.1611,-0.22507,-0.14132,-0.13031,	0.042038,0.29175,0.196992,0.195642,	0.341204,0.115252,0.121781]

#A19
#cst_weights = [-0.1508,	-0.17312,	-0.17365,	-0.12959,	-0.15798,	-0.15435,	0.163213,	0.148695,	0.149951,	0.115497,	0.157708,	0.130644]

#B64
#cst_weights = [-0.13244,	-0.01438,	-0.11074,	0.013883,	-0.07975,	-0.01922,	0.193961,	0.339016,	0.115781,	0.417358,	0.158287,	0.301708]

#NACA 2411
#cst_weights = [-0.13512611, -0.07832579, -0.06839925, -0.06866104, -0.04561013, -0.05078831,0.18301484, 0.20030699, 0.23139451, 0.17648309, 0.22436628, 0.21250357]

cst_weights = [-0.15029182, -0.16098318, -0.20377234, -0.13863304, -0.13868888,  0.04034309, 0.30174562, 0.21011278, 0.17247788, 0.31289289 ,0.1346193 , 0.12938747]


def predict_xgb(alphas):
    """
    Predict CL and CD for multiple alpha values using XGBoost
    """
    # Convert to list if single value
    if isinstance(alphas, (int, float)):
        alphas = [alphas]

    n = len(alphas)

    # Create DataFrame with repeated weights
    data_dict = {}
    for i, col in enumerate(['w_l1', 'w_l2', 'w_l3', 'w_l4', 'w_l5', 'w_l6',
                             'w_u1', 'w_u2', 'w_u3', 'w_u4', 'w_u5', 'w_u6']):
        data_dict[col] = [cst_weights[i]] * n

    data = pd.DataFrame(data_dict)
    data['Alpha'] = alphas

    # Predict (no scaling needed for XGBoost)
    cl = xgb_cl.predict(data)
    cd = xgb_cd.predict(data)

    return cl, cd


# Use it
alphas = [-2, 0, 2, 4, 6, 8, 10]
cl_values, cd_values = predict_xgb(alphas)

# Print results with formatting
print("\nXGBoost Results:")
print("-" * 50)
for alpha, cl, cd in zip(alphas, cl_values, cd_values):
    print(f"Alpha = {alpha:3d}°  |  CL = {cl:.4f}  |  CD = {cd:.6f}")
print("-" * 50)