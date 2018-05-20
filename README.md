# MakineOgrenmesi
//
 logistic = LogisticRegression()
 classifier = DecisionTreeClassifier()
 lm=linear_model.LinearRegression()
 lm.fit(..,..)
 lm.coef_.tolist()
 X_train, X_test, y_train, y_test = train_test_split(..,.. , test_size=0.2,  random_state=1)
 
 reg = linear_model.LinearRegression()
 reg.fit(X_train, y_train)

 logistic.fit(X_train, y_train)
 classifier.fit(X_train, y_train)

 print('Coefficients: \n', reg.coef_)
 print('Variance score: {}'.format(reg.score(X_test, y_test)))
 modelin_tahmin_ettigi_y = reg.predict(tahmin)
 y_pred = classifier.predict(X_test)
 logic_predic=logistic.predict(tahmin)
 score = logistic.score(X_test, y_test)

 print("Decision Tree")
 print(classification_report(y_test, y_pred))
 df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
 print(df.head(20))
 print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
 print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


 print(" LogisticRegression ")
 print("Logic Accuracy Score = % "+str(score*100))

 decision_predict = classifier.predict(tahmin)
//
