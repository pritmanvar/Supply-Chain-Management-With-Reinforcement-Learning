import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import pickle
import ray


def get_demand(t, date="2024-4-11"):
    date_object = datetime.strptime(date, '%Y-%m-%d')
    date = str(date_object+timedelta(days=int(t)))[:10]
    with open("prophet_model.pkl", 'rb') as f:
        m = pickle.load(f)
        future_dates = pd.DataFrame({'ds': [date]})
        forecast = m.predict(future_dates)
        return forecast['yhat'][0]/100


def get_storage_insites():
    # load model
    loaded_ppo = Algorithm.from_checkpoint('./checkpoint_000000')
    loaded_policy = loaded_ppo.get_policy()
    ray.shutdown()
    ray.init()

    date = input("Enter the date in this yyyy-mm-dd format:  ")
    manufacturing_price = int(input("Enter the Manufacturing Price : "))
    selling_price = int(input("Enter the Selling price: "))

    current_stock = 0
    selling_price = 100
    manufacturing_price = 70
    storage_cost = 100
    capacity = 1000

    profit_list = []
    current_stock_list = []
    new_stock_list = []
    demand_list = []
    production_cost_list = []
    revenue_list = []

    for i in range(15):
        demand_history = np.array([get_demand(i+time, date)
                                  for time in range(15)])
        new_stock = loaded_ppo.compute_single_action(
            [np.hstack((demand_history, np.array(current_stock)))])[0]
        if new_stock + current_stock > capacity:
            new_stock = capacity - current_stock

        total_revenue = selling_price * min(current_stock, demand_history[0])
        total_production_cost = manufacturing_price * new_stock
        total_storage_cost = storage_cost
        profit = total_revenue - total_production_cost - storage_cost

        current_stock_list.append(current_stock)
        new_stock_list.append(new_stock)
        demand_list.append(demand_history[0])
        production_cost_list.append(total_production_cost)
        profit_list.append(profit)
        revenue_list.append(total_revenue)

        print()
        print("Day number:", i+1)
        print("Current stock is:", current_stock)
        print("agent requests for {} this much products".format(new_stock))
        print("demand for next 15 days is:", demand_history)
        print("manufucturing cost:", total_production_cost)
        print("Storage cost:", storage_cost)
        print("Profit:", profit)
        print()

        current_stock = new_stock + current_stock - \
            min(current_stock, demand_history[0])

    fig, axs = plt.subplots(3, figsize=(16, 12))

    axs[0].plot([i for i in range(len(current_stock_list))],
                current_stock_list, label='Current Stock')
    axs[0].plot([i for i in range(len(new_stock_list))],
                new_stock_list, label='New Stock')
    axs[0].plot([i for i in range(len(demand_list))],
                demand_list, label='Demand')
    axs[0].set_title('Current Stock, New Stock and Demand over Time')
    axs[0].legend()

    axs[1].plot([i for i in range(len(np.cumsum(production_cost_list)))], np.cumsum(
        production_cost_list), label='Total production cost')
    axs[1].plot([i for i in range(len(np.cumsum(revenue_list)))],
                np.cumsum(revenue_list), label='Total revenue')
    axs[1].set_title('Production Cost and Revenue Generated over Time')
    axs[1].legend()

    axs[2].plot([i for i in range(len(np.cumsum(profit_list)))],
                np.cumsum(profit_list), label='Total Profit')
    axs[2].set_xlabel('Days')
    axs[2].set_title('Profit Generated over Time')
    axs[2].legend()

    return plt


def main():
    plot = get_storage_insites()
    plot.show()


if __name__ == "__main__":
    main()
