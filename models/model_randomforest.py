import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Ignore warnings
warnings.filterwarnings('ignore')


class RandomForestSpreadModel:
    
    def __init__(self, n_estimators=100, max_depth=None, random_state = 42):
        """
        Initialize the Random Forest model with specified parameters.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.target_name = 'spread_close_pct'
        self.is_fitted = False
              
    def load_data(self,symbol):
        
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / 'featured_data'
        file_path = data_path / f'featured_{symbol}_data.csv'
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        
    def prepare_features(self, exclude_features=None):
        
        default_exclude = [
        'time', 
        self.target_name,
        'spread_close_absolute',  # Direct calculation from target
        'is_opportunity',  # Target-related
        'is_opportunity_flag',  # Target-related
        'is_real_opportunity',  # Target-related
        'buy_exchange',  # Categorical
        'sell_exchange',  # Categorical
        'buy_exchange_lag_1',  # Categorical
        'sell_exchange_lag_1',  # Categorical
        'high_exchange',  # Categorical
        'low_exchange',  # Categorical
        'min_close',  # Used to calculate target
        'max_close',  # Used to calculate target
        'price_ratio_buy_sell',  # Directly related to spread
        'opportunity_gap',  # Directly related to spread
            # These features are derived from spread and cause perfect prediction
            'spread_diff_from_lag_1',  # Derived from spread
            'spread_diff_from_lag_5',  # Derived from spread
            'spread_rate_change',  # Derived from spread
            'spread_rate_change_pct',  # Derived from spread
            'spread_rate_acceleration'  # Derived from spread
        ]

        # Verify labels
        X = self.df.drop(columns=default_exclude, errors = 'ignore')
        self.feature_names = X.columns.tolist()
        y = self.df[self.target_name]
        
        
        # 80/20 split
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]

    def train(self):
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared yet, use prepare_features() first.")
        
        
        # Train the model and change the fitted flag
        print("Training Random Forest Model")
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        
        
        #Training score
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"Training R² Score: {train_score:.4f}")
        
        
    def predict(self,X):
        
        if not self.is_fitted:
            raise ValueError("Model not trained yet, use train() first.")
                             
        predictions = self.model.predict(X)
        
        return predictions
    
    
    def evaluate(self):
        
        y_pred = self.predict(self.X_test)
        
        MSE = mean_squared_error(self.y_test, y_pred)
        MAE = mean_absolute_error(self.y_test, y_pred)
        R2 = r2_score(self.y_test, y_pred)
        
        print(f"MSE: {MSE:.4f}")
        print(f"MAE: {MAE:.4f}")
        print(f"R² Score: {R2:.4f}")
        
        return MSE, MAE, R2
    
    
    def plot_results(self, y_test, y_pred, save_path = None):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        residuals = y_test - y_pred
        abs_error = np.abs(residuals)

        #axes[0, 0] - Actual Vs predicted 
        axes[0 ,0].scatter(y_test, y_pred, alpha = 0.5, s = 10)
        axes[0 ,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2, label = 'Perfect Prediction')
        axes[0 ,0].set_xlabel('Actual spread_close_pct', fontsize = 12)
        axes[0 ,0].set_ylabel('Predicted spread_close_pct', fontsize = 12)
        axes[0 ,0].set_title('Actual vs Predicted', fontsize = 14, fontweight = 'bold')
        axes[0 ,0].legend()
        axes[0 ,0].grid(True, alpha = 0.3)
        
        #axes[0, 1] - Residuals
        axes[0 ,1].scatter(y_pred, residuals, alpha = 0.5, s = 10)
        axes[0 ,1].axhline(y = 0, color = 'r', linestyle = '--', lw = 2)
        axes[0 ,1].set_xlabel('Predicted spread_close_pct', fontsize = 12)
        axes[0 ,1].set_ylabel('Residuals', fontsize = 12)
        axes[0 ,1].set_title('Residual Plot', fontsize = 14, fontweight = 'bold')
        axes[0 ,1].grid(True, alpha = 0.3)
        
        #axes[1, 0] - Residual Distribution
        axes[1 ,0].hist(residuals, bins = 50, edgecolor = 'black', alpha = 0.7)
        axes[1 ,0].axvline(x = 0, color = 'r', linestyle = '--', lw = 2)
        axes[1 ,0].set_xlabel('Residuals', fontsize = 12)
        axes[1 ,0].set_ylabel('Frequency', fontsize = 12)
        axes[1 ,0].set_title('Residual Distribution', fontsize = 14, fontweight = 'bold')
        axes[1 ,0].grid(True, alpha = 0.3)
        
        #axes[1, 1] - Absolute Error Distribution
        axes[1 ,1].hist(abs_error, bins = 50, edgecolor = 'black', alpha = 0.7, color = 'orange')
        axes[1 ,1].set_xlabel('Absolute Error', fontsize = 12)
        axes[1 ,1].set_ylabel('Frequency', fontsize = 12)
        axes[1 ,1].set_title('Absolute Error Distribution', fontsize = 14, fontweight = 'bold')
        axes[1 ,1].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Results plot saved to {save_path}")
        else:
            plt.show()  
            
        plt.close()
        
        
    def plot_feature_importance(self, top_n = 20, save_path = None):
        
        # Get feature importance and top features from the model
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        importance_df = importance_df.sort_values(by = 'importance', ascending = False)
        top_features = importance_df.head(top_n)
        
        
        ## Plot ##
        
        fig, ax = plt.subplots(figsize = (12, 8))
        
        # Create horizontal bar chart
        ax.barh(range(len(top_features)), top_features['importance'], color = 'steelblue', alpha = 0.7)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance', fontsize = 12)
        ax.set_ylabel('Features', fontsize = 12)
        ax.set_title(f'Top {top_n} Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha = 0.3, axis='x')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        else:
            plt.show()
            
        plt.close()
        
        

    def get_feature_importance(self, top_n=20):
    
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        top_features = importance_df.head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(top_features.to_string(index=False))
        
        return top_features
        
                
        
    def plot_prediction_hist(self, y_pred, save_path = None):

        plt.figure(figsize = (10, 6))
        plt.hist(y_pred, bins = 40, edgecolor = 'black', alpha = 0.75)
        plt.xlabel('Predicted spread_close_pct', fontsize = 12)
        plt.ylabel('Frequency', fontsize = 12)
        plt.title('Prediction Histogram', fontsize = 14, fontweight = 'bold')
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Prediction histogram saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        
        
def main():
    
    np.random.seed(42)
    
    base_path = Path(__file__).parent.parent
    
    crypto = 'BTCUSD' ## TO BE CHANGED LATER
    
    # Model initialization
    print(f"\n{'='*60}")
    print(f"Initializing Random Forest Model for {crypto}")
    print(f"{'='*60}\n")
    
    model = RandomForestSpreadModel(n_estimators=100, max_depth=20, random_state = 42)
    
    # Load and prepare data
    model.load_data(crypto)
    model.prepare_features()
    
    # Train the model
    model.train()
    
    # Evaluate the model
    model.evaluate()
    
    # Feature importance
    model.get_feature_importance(top_n=20)
    
    # Make predictions
    y_pred = model.predict(model.X_test)
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'random-forest' / crypto
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    model.plot_results(model.y_test, y_pred, save_path=output_dir / 'results.png')
    model.plot_feature_importance(top_n=20, save_path=output_dir / 'feature_importance.png')
    model.plot_prediction_hist(y_pred, save_path=output_dir / 'prediction_hist.png')
    
    
    print(f"\n{'='*60}")
    print(f"Random Forest Model for {crypto} Completed")
    print(f"{'='*60}\n")
    
if __name__ == "__main__":
        main()  