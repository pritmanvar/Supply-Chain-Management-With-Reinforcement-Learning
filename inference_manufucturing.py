import ray
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import pandas as pd
import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from inference_storage import get_demand
from transportation import get_locationiq_routes,Transportation


class ManufactoringState(object):
    def __init__(self, manufacturing_rate=10, current_stock_raw_material=10, current_product_stock=0, demand_history=[], raw_material_history=[]):
        self.manufacturing_rate = manufacturing_rate
        self.current_stock_raw_material = current_stock_raw_material
        self.current_product_stock = current_product_stock
        self.demand_history = demand_history
        self.raw_material_history = raw_material_history
        self.time = 0

    def to_array(self, ):
        return np.hstack((np.array(self.manufacturing_rate), np.array(self.current_product_stock), np.array(self.current_stock_raw_material), np.array(self.demand_history), np.array(self.raw_material_history)))
    
def get_manufucturing_insites():
    #load model 
    loaded_ppo_manufucturing = Algorithm.from_checkpoint('./checkpoint_manufucturing')
    # ray.shutdown()
    # ray.init()

    date = input("Enter the date in this yyyy-mm-dd format:  ")
    raw_material_cost = int(input("Enter the raw material cost: "))
    main_cost = int(input("Enter the sum of storage cost and manufacturing cost per day: "))
    production_cost_per_product = int(input("Enter the production cost per product: "))
    max_manufacturing_rate = int(input("Enter the max manufucturing rate per hour: "))
    product_capacity = int(input("Enter the manufuctured product capacity: "))
    selling_price = int(input("Enter the selling price of the product: "))
    raw_material_capacity = int(input("Enter the raw material capacity: "))
    require_raw_material_per_product = int(input("Enter the required raw material per product: "))

    #get source and destination address
    vehicals_capacity = str(input("Enter capacity of your unique vehicles seperated by space. -> capacity_for_v1 capacity_for_v2 ")).split(" ")
    vehicals_cost = str(input("Enter cost of each vehical per KM seperated by space. -> cost_for_v1 cost_for_v2 ")).split(" ")
    num_of_vehicals_per_type = str(input("Enter number of vehicals instance for each type seperated by space. -> count_for_v1 count_for_v2 ")).split(" ")
    soucre_address = str(input("Enter source address: "))
    destination_address = str(input("Enter destination address: "))
    
    try:
        routes_info = get_locationiq_routes(source=soucre_address,destination= destination_address)
        routes_km = routes_info['distance']
        routes_time = routes_info['duration']
    except Exception as e:
        print(e)
        return None
    
    vehicals_capacity = [int(x) for x in vehicals_capacity]
    vehicals_cost = [int(x) for x in vehicals_cost]
    num_of_vehicals_per_type = [int(x) for x in num_of_vehicals_per_type]
    routes_km = [int(x) for x in routes_km]
    routes_time = [int(x) for x in routes_time]

    state = ManufactoringState()
    state.demand_history = [get_demand(time, date) for time in range(15)]
    state.raw_material_history = [0 for i in range(15)]

    profit_list = []
    current_product_stock_list = []
    current_raw_material_stock_list = []
    manufactured_product_list = []
    new_raw_material_list = []
    manufacturing_rate_list = []
    total_products_delivered_list = []
    demand_list = []
    production_cost_list = []
    revenue_list = []

    for i in range(15):
        current_state = state.to_array()
        action = loaded_ppo_manufucturing.compute_single_action(current_state)
        new_manufacturing_rate, new_raw_material_stock = action

        #calculate transportation time :
        transportation_obj  = Transportation(int(np.floor(state.demand_history[0])),vehicals_capacity,vehicals_cost,num_of_vehicals_per_type,routes_km,routes_time,0.5)
        transportation_cost,transportation_time = transportation_obj.inference()
        transportation_time /=60
        if new_manufacturing_rate > max_manufacturing_rate:
            new_manufacturing_rate = max_manufacturing_rate

        new_possible_stock = new_manufacturing_rate * 24

        # how much we can actually produce with the raw material
        if new_possible_stock*require_raw_material_per_product > state.current_stock_raw_material:
            new_possible_stock = np.floor(
                state.current_stock_raw_material/require_raw_material_per_product)
            
        if new_possible_stock+state.current_product_stock > product_capacity:
            new_possible_stock = product_capacity - state.current_product_stock
            
        new_manufacturing_rate = new_possible_stock/(24-transportation_time)
            
        # how much raw material we can store
        if (state.current_stock_raw_material + new_raw_material_stock - (new_possible_stock*require_raw_material_per_product)) > raw_material_capacity:
            new_raw_material_stock = raw_material_capacity - state.current_stock_raw_material + \
                (new_possible_stock*require_raw_material_per_product)

        # what can be fullfiled with this time limit
        if (24 - transportation_time) < new_possible_stock/new_manufacturing_rate:
            new_possible_stock = (
                24-transportation_time)*new_manufacturing_rate
        total_products_to_deliver = min(
            new_possible_stock+state.current_product_stock, state.demand_history[0])

        revenue = selling_price*total_products_to_deliver
        total_costs = (production_cost_per_product * new_possible_stock) + main_cost + \
            (raw_material_cost * new_raw_material_stock) + transportation_cost

        current_product_stock_list.append(state.current_product_stock)
        current_raw_material_stock_list.append(state.current_stock_raw_material)
        manufactured_product_list.append(new_possible_stock)
        new_raw_material_list.append(new_raw_material_stock)
        manufacturing_rate_list.append(new_manufacturing_rate)
        total_products_delivered_list.append(total_products_to_deliver)
        demand_list.append(state.demand_history[0])
        production_cost_list.append(total_costs)
        revenue_list.append(revenue)
        reward = revenue - total_costs
        previous_demands = state.demand_history[0:14]
        state.demand_history[0] = get_demand(15+i)
        state.demand_history[1:15] = previous_demands

        previous_raw_history = state.raw_material_history[0:14]
        state.raw_material_history[0] = new_raw_material_stock
        state.raw_material_history[1:15] = previous_raw_history
        state = ManufactoringState(new_manufacturing_rate, state.current_stock_raw_material + new_raw_material_stock - (new_possible_stock*require_raw_material_per_product),
                                (new_possible_stock + state.current_product_stock - total_products_to_deliver), state.demand_history, state.raw_material_history)

        profit_list.append(reward)
        print()
        print("Day number:", i+1)
        print("Current stock is:", state.current_product_stock)
        print("agent requests for {} this much products".format(new_possible_stock))
        print("demand for next 15 days is:", state.demand_history)
        print("manufucturing cost:", total_costs)
        print("Profit:", reward)
        print()
        
    fig, axs = plt.subplots(5, figsize=(16, 12))

    axs[0].plot([i for i in range(len(current_product_stock_list))], current_product_stock_list, label='Current Stock')
    axs[0].plot([i for i in range(len(manufactured_product_list))], manufactured_product_list, label='New Stock')
    axs[0].plot([i for i in range(len(demand_list))], demand_list, label='Demand')
    axs[0].set_title('Current Stock, New Stock and Demand over Time')
    axs[0].legend()

    axs[1].plot([i for i in range(len(np.cumsum(production_cost_list)))], np.cumsum(production_cost_list), label='Total production cost')
    axs[1].plot([i for i in range(len(np.cumsum(revenue_list)))], np.cumsum(revenue_list), label='Total revenue')
    axs[1].set_title('Production Cost and Revenue Generated over Time')
    axs[1].legend()

    axs[2].plot([i for i in range(len(np.cumsum(profit_list)))], np.cumsum(profit_list), label='Total Profit')
    axs[2].set_xlabel('Days')
    axs[2].set_title('Profit Generated over Time')
    axs[2].legend()


    axs[3].plot([i for i in range(len((new_raw_material_list)))], (new_raw_material_list), label='Total production cost')
    axs[3].plot([i for i in range(len((current_raw_material_stock_list)))], (current_raw_material_stock_list), label='Total revenue')
    axs[3].set_title('New Raw material stock and current_raw material stock over Time')
    axs[3].legend()

    axs[4].plot([i for i in range(len((manufacturing_rate_list)))], (manufacturing_rate_list), label='Total Profit')
    axs[4].set_xlabel('Days')
    axs[4].set_title('Manufacturing Rate over time')
    axs[4].legend()

    return plt

# def main(): 
#     plot = get_manufucturing_insites()
#     if plot is not None:
#         plot.show()
  
# if __name__=="__main__": 
#     main() 