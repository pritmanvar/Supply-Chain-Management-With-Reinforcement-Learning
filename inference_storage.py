import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import pickle
import pandas

from inference_manufucturing_2 import get_manufucturing_insights

def get_demand(t, date="2024-4-11"):
    date_object = datetime.strptime(date, '%Y-%m-%d')
    date = str(date_object+timedelta(days=int(t)))[:10]
    with open("prophet_model.pkl", 'rb') as f:
        m = pickle.load(f)
        future_dates = pd.DataFrame({'ds': [date]})
        forecast = m.predict(future_dates)
        return forecast['yhat'][0]//100

df = pd.DataFrame(columns=['date', 'current_stock', 'new_stock', 'demand', 'production_cost', 'revenue', 'profit'])
# profit_list = []
# current_stock_list = []
# new_stock_list = []
# demand_list = []
# production_cost_list = []
# revenue_list = []

def get_storage_insights(date:str="2024-04-15", manufacturing_price:int=70, selling_price:int=100, capacity:int=1000, storage_cost: int = 100, manufucturing_data: dict = {}):
    loaded_ppo = Algorithm.from_checkpoint('./checkpoint_000000')
    print(manufucturing_data)
    # loaded_policy = loaded_ppo.get_policy()
    # ray.shutdown()
    # ray.init()

    # date = input("Enter the date in this yyyy-mm-dd format:  ")
    # manufacturing_price = int(input("Enter the Manufacturing Price : "))
    # selling_price = int(input("Enter the Selling price: "))

    current_stock = 0

    global df
    
    for i in range(15):
        demand_history = np.array([get_demand(i+time, date) for time in range(15)])
        new_stock = loaded_ppo.compute_single_action([np.hstack((demand_history, np.array(current_stock)))])[0]
        if new_stock + current_stock - min(current_stock, demand_history[0]) > capacity:
            new_stock = capacity - current_stock + min(current_stock, demand_history[0])

        new_stock = int(get_manufucturing_insights(manufucturing_data["raw_material_cost"], manufucturing_data["main_cost"],manufucturing_data['production_cost_per_product'],manufucturing_data['max_manufacturing_rate'],manufucturing_data['product_capacity'],manufucturing_data['selling_price'],manufucturing_data['raw_material_capacity'],manufucturing_data['require_raw_material_per_product'],manufucturing_data['vehicals_capacity'],manufucturing_data['vehicals_cost'],manufucturing_data['num_of_vehicals_per_type'],manufucturing_data['source_address'],manufucturing_data['destination_address'], new_stock, date ))
        total_revenue = selling_price * min(current_stock, demand_history[0])
        total_production_cost = manufacturing_price * new_stock
        profit = total_revenue - total_production_cost - storage_cost

        # current_stock_list.append(current_stock)
        # new_stock_list.append(new_stock)
        # demand_list.append(demand_history[0])
        # production_cost_list.append(total_production_cost)
        # profit_list.append(profit)
        # revenue_list.append(total_revenue)

        df.loc[len(df)] = [str(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=int(i)))[:10], current_stock, new_stock, demand_history[0], total_production_cost, total_revenue, profit]


        print()
        print("Day number: storage ", i+1)
        print("Current stock is:", current_stock)
        print("agent requests for {} this much products".format(new_stock))
        print("demand for next 15 days is:", demand_history)
        print("manufucturing cost:", total_production_cost)
        print("Storage cost:", storage_cost)
        print("Profit:", profit)
        print()

        current_stock = new_stock + current_stock - min(current_stock, demand_history[0])
            
def get_storage_graphs():
    global df
    fig, axs = plt.subplots(3, figsize=(16, 12))

    axs[0].plot([i for i in range(len(df['current_stock']))], df['current_stock'], label='Current Stock')
    axs[0].plot([i for i in range(len(df['new_stock']))], df['new_stock'], label='New Stock')
    axs[0].plot([i for i in range(len(df['demand']))], df['demand'], label='Demand')
    axs[0].get_xaxis().set_visible(False)
    axs[0].set_title('Current Stock, New Stock and Demand over Time')
    axs[0].legend()

    axs[1].plot([i for i in range(len(np.cumsum(df['production_cost'])))], np.cumsum(df['production_cost']), label='Total production cost')
    axs[1].plot([i for i in range(len(np.cumsum(df['revenue'])))], np.cumsum(df['revenue']), label='Total revenue')
    axs[1].get_xaxis().set_visible(False)
    axs[1].set_title('Production Cost and Revenue Generated over Time')
    axs[1].legend()

    axs[2].plot([i for i in range(len(np.cumsum(df['profit'])))], np.cumsum(df['profit']), label='Total Profit')
    axs[2].set_xlabel('Days')
    axs[2].set_title('Profit Generated over Time')
    axs[2].legend()
    
    df.to_csv("Storage_details.csv")


    return plt


def main():
    plot = get_storage_insights()
    plot.show()


if __name__ == "__main__":
    main()
