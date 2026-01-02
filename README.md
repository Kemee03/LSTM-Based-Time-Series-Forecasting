# **Apple (AAPL) Closing Price Forecast using Univariate LSTM**



### **Overview**

The project uses a **univariate LSTM model** to forecast daily closing prices of Apple (AAPL) stock. The goal was to design a clean, realistic and reproducible forecasting pipeline from raw data parsing to model evaluation while keeping temporal integrity intact.



### **Features**

* Dataset: Real historical **AAPL stock data from Yahoo Finance**
* **Sliding-window sequence** generation (30-day lookback)
* **LSTM architecture** implemented via **TensorFlow-Keras** backend
* Evaluation on final 20% chronological hold-out period using **RMSE \& MAE**
* **Prediction plots** and **training loss curves**
* Both **raw and cleaned datasets** included for transparency



### **Tech Stack**

<table> <tr> <th>Stage</th> <th>Tool Used</th> </tr> <tr> <td>Deep Learning Model</td> <td>TensorFlow-Keras (LSTM, Dense, Dropout)</td> </tr> <tr> <td>Data Processing</td> <td>Pandas, NumPy, Scikit-Learn (MinMaxScaler)</td> </tr> <tr> <td>Evaluation</td> <td>RMSE, MAE (math + sklearn)</td> </tr> <tr> <td>Visualization</td> <td>Matplotlib</td> </tr> <tr> <td>Runtime</td> <td>Python 3.x / Jupyter / Google Colab</td> </tr> <tr> <td>Version Control</td> <td>Git + GitHub</td> </tr> </table>



### **Files Included**

<table> <tr> <th>File</th> <th>Purpose</th> </tr> <tr> <td><code>time\_series\_data.txt</code></td> <td>Raw AAPL dataset downloaded from Yahoo Finance</td> </tr> <tr> <td><code>stock\_prices.csv</code></td> <td>Cleaned dataset used for LSTM training</td> </tr> <tr> <td><code>time\_series\_forecasting.ipynb</code></td> <td>Notebook for parsing, preprocessing, training, prediction, and evaluation</td> </tr> <tr> <td><code>README.md</code></td> <td>Project documentation</td> </tr> </table>



### **How to Run**

<table>

&nbsp; <tr>

&nbsp;   <th>Step</th>

&nbsp;   <th>Action / Command</th>

&nbsp; </tr>



&nbsp; <tr>

&nbsp;   <td><b>1.</b> Install dependencies</td>

&nbsp;   <td>

&nbsp;     <code>pip install pandas numpy matplotlib tensorflow scikit-learn</code>

&nbsp;     <ul>

&nbsp;       <li>Run this command in your system shell or Git Bash terminal ✔</li>

&nbsp;     </ul>

&nbsp;   </td>

&nbsp; </tr>



&nbsp; <tr>

&nbsp;   <td><b>2.</b> Dataset check</td>

&nbsp;   <td>Ensure dataset contains a numeric <code>Close</code> column and is chronological</td>

&nbsp; </tr>



&nbsp; <tr>

&nbsp;   <td><b>3.</b> Run notebook</td>

&nbsp;   <td>Open and run <code>time\_series\_forecasting.ipynb</code> cells sequentially (top-to-bottom)</td>

&nbsp; </tr>



&nbsp; <tr>

&nbsp;   <td><b>4.</b> Reproducibility</td>

&nbsp;   <td>For reproducibility, keep <code>window = 30</code> and avoid shuffling during train/test split</td>

&nbsp; </tr>

</table>





### **Evaluation Results** 

<table> <tr> <th>Metric</th> <th>Result</th> </tr> <tr> <td>Mean Absolute Error (MAE)</td> <td>~0.63</td> </tr> <tr> <td>Root Mean Squared Error (RMSE)</td> <td>~0.83</td> </tr> </table>



(RMSE > MAE is expected as RMSE penalizes larger errors more)



### **Key Notes** 

* Used a real financial dataset, not a demo sample
* Avoided train/test shuffling to preserve temporal structure
* Fixed parsing issues manually before training
* Scaler leakage removed to maintain fairness in evaluation
* Raw and cleaned data both committed to show real-world data handling experience



### **Future Scope**

<table> <tr><td>• Attention-based LSTM</td></tr> <tr><td>• Uncertainty estimation (Monte-Carlo Dropout)</td></tr> <tr><td>• Dockerized FastAPI deployment + CI/CD</td></tr> <tr><td>• Multivariate extension using OHLC + Volume</td></tr> </table>

