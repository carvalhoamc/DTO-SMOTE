# DTO-SMOTE
Delaunay Tesselation Oversampling SMOTE.

Run DTO-SMOTE Biclass Datasets :
#python3 main.py

-----------------------------------------------
Run DTO-SMOTE Multiclass Datasets:

File: datasetsDelaunay.py line 82
Change 
datasets_list3_MultClasses
To
datasets

File: oversampling.py

Comment line 159: 
Y_train = self.converteY(Y_train)  # Biclass only

Comment line 184:
df.at[i, 'AUC'] = roc_auc_score(Y_test, Y_pred)  # biclass

comment line 221:
df.at[i, 'AUC'] = roc_auc_score(Y_test, Y_pred)  # biclass

comment line 225:
df.to_csv(output_dir + 'results_biclass_' + p.__class__.__name__ + '.csv', index=False)

uncoment line 226:
df.to_csv(output_dir + 'results_multiclass_' + p.__class__.__name__ + '.csv', index=False)

After these run:
#python3 main.py




