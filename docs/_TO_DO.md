2. Test how the window size impacts the performance of the model when predicting for 1, 5, 10, 20, 60 (3mo), 250 (1y) days.

3. Most used technical indicators(as per paper, page 6) : 
    RSI, 
    stochastic oscilator, 
    moving average convergence-divergence,
    Altman Z-score,
    Yield curve and its inversion ( use 3y vs 10y),
    50 and 150 ma,
    Oversold signal - Larry Williams - Spx oversold signal,

4. Most used fundamental indicators as per paper(page 6):
    Overall economic status of the country,
    microeconomic factors (company info),
    macroeconomic factors ( interest rates, inflation, CPI, PPI, unemployment rate, etc.),
    See Sven's book as well as Book on stocks from Alex.
    
5. Most used feature selection technics to try with the model (read more specifics on the paper) with citations to other papers: 
    RF, (Try Recursive Feature Elimination with Cross-Validation. In sklearn if possible. read.)
    PCA, (https://plotly.com/python/pca-visualization/),
    Autoencoder,
    Ansamble of Fearute selections ( also read in paper if needed),

6. Find literature and test myself on buckets of stocks what fundamental indicators are most important fro predicting stock price. Use them as features in the model.

6. Consider spliting dataset randomly into sliding windows for training set, then train the model and evaluate on the test set. for different test sets (like 300 days * 5 times), and then average the performance. See more at the paper (https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00441-7) page 21.

7. Incorporate analysis of analysts' recommendations. WHere to find this data ? what time horizon each recommendation is for ? How to incorporate this data into the model ? LLM to look for time horizon or API has it braken down ? 

