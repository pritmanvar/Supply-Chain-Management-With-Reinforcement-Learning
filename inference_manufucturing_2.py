import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from transportation import get_locationiq_routes,Transportation
from datetime import datetime, timedelta

class ManufactoringState(object):
    def __init__(self, manufacturing_rate=0, current_stock_raw_material=0, current_product_stock=0, demand_history=[0 for i in range(15)], raw_material_history=[0 for i in range(15)]):
        self.manufacturing_rate = manufacturing_rate
        self.current_stock_raw_material = current_stock_raw_material
        self.current_product_stock = current_product_stock
        self.demand_history = demand_history
        self.raw_material_history = raw_material_history
        self.time = 0

    def to_array(self, ):
        return np.hstack((np.array(self.manufacturing_rate), np.array(self.current_product_stock), np.array(self.current_stock_raw_material), np.array(self.demand_history), np.array(self.raw_material_history)))

loaded_ppo_manufucturing = Algorithm.from_checkpoint('./checkpoint_manufucturing')
# ray.shutdown()
# ray.init()

# profit_list = []
# current_product_stock_list = []
# current_raw_material_stock_list = []
# manufactured_product_list = []
# new_raw_material_list = []
# manufacturing_rate_list = []
# total_products_delivered_list = []
# demand_list = []
# production_cost_list = []
# revenue_list = []
routes_km = []
routes_time = []
day_number = 0
df = pd.DataFrame(columns=['date', 'current_stock_of_products', 'new_stock_of_products', 'demand', 'current_stock_of_raw_material', 'new_stock_of_raw_material', 'manufacturing_rate', 'total_products_delivered', 'production_cost', 'transpotation_cost', 'transpotation_time', 'revenue', 'profit'])


state = ManufactoringState()
def get_routes_info(source: str, destination: str) -> tuple:
    try:
        routes_info = get_locationiq_routes(source=source,destination= destination)
        routes_km = routes_info['distance']
        routes_time = routes_info['duration']
        routes_km = [int(x) for x in routes_km]
        routes_time = [int(x) for x in routes_time]
        
        return routes_km, routes_time
    except Exception as e:
        print(e)
        return None, None
    
def get_manufucturing_insights(raw_material_cost, main_cost, production_cost_per_product, max_manufacturing_rate, product_capacity, selling_price, raw_material_capacity, require_raw_material_per_product, vehicals_capacity, vehicals_cost, num_of_vehicals_per_type, source_address, destination_address, demand, date):
    vehicals_capacity = [int(x) for x in vehicals_capacity]
    vehicals_cost = [int(x) for x in vehicals_cost]
    num_of_vehicals_per_type = [int(x) for x in num_of_vehicals_per_type]

    global state, loaded_ppo_manufucturing, day_number, routes_km, routes_time, df

    if routes_km is None or len(routes_km) == 0:
        routes_km, routes_time = get_routes_info(source_address, destination_address)

    current_state = state.to_array()
    action = loaded_ppo_manufucturing.compute_single_action(current_state)
    new_manufacturing_rate, new_raw_material_stock = int(action[0]), int(action[1])    

    print(routes_km, "ROUTES KM")
    #calculate transportation time :
    transportation_obj  = Transportation(int(np.floor(demand)),vehicals_capacity,vehicals_cost,num_of_vehicals_per_type,routes_km,routes_time,0.5)
    transpotation_cost,transpotation_time = transportation_obj.inference()
    transpotation_time //=60
    
    print(transpotation_cost, transpotation_time, "Transpotation details")
    
    if new_manufacturing_rate > max_manufacturing_rate:
        new_manufacturing_rate = max_manufacturing_rate

    new_possible_stock = new_manufacturing_rate * 24

    # how much we can actually produce with the raw material
    if new_possible_stock*require_raw_material_per_product > state.current_stock_raw_material:
        new_possible_stock = state.current_stock_raw_material//require_raw_material_per_product
        
    if new_possible_stock+state.current_product_stock > product_capacity:
        new_possible_stock = product_capacity - state.current_product_stock
        
    new_manufacturing_rate = new_possible_stock//(24-transpotation_time) if transpotation_time < 24 else 0
            
    # how much raw material we can store
    if (state.current_stock_raw_material + new_raw_material_stock - (new_possible_stock*require_raw_material_per_product)) > raw_material_capacity:
        new_raw_material_stock = raw_material_capacity - state.current_stock_raw_material + (new_possible_stock*require_raw_material_per_product)

    # what can be fullfiled with this time limit
    
    if new_manufacturing_rate > 0 and (24 - transpotation_time) < new_possible_stock//new_manufacturing_rate:
        new_possible_stock = (24-transpotation_time)*new_manufacturing_rate
        
    total_products_to_deliver = min(new_possible_stock+state.current_product_stock, demand)

    revenue = selling_price*total_products_to_deliver
    total_costs = (production_cost_per_product * new_possible_stock) + main_cost + (raw_material_cost * new_raw_material_stock) + transpotation_cost

    reward = revenue - total_costs
    print("Total delivered products", total_products_to_deliver)
    
    # current_product_stock_list.append(state.current_product_stock)
    # current_raw_material_stock_list.append(state.current_stock_raw_material)
    # manufactured_product_list.append(new_possible_stock)
    # new_raw_material_list.append(new_raw_material_stock)
    # manufacturing_rate_list.append(new_manufacturing_rate)
    # total_products_delivered_list.append(total_products_to_deliver)
    # demand_list.append(state.demand_history[0])
    # production_cost_list.append(total_costs)
    # revenue_list.append(revenue)
    previous_demands = state.demand_history[0:14]
    state.demand_history[0] = demand
    state.demand_history[1:15] = previous_demands

    df.loc[len(df)] = [str(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=int(day_number)))[:10], state.current_product_stock, new_possible_stock, state.demand_history[0], state.current_stock_raw_material, new_raw_material_stock, new_manufacturing_rate, total_products_to_deliver, total_costs, transpotation_cost, transpotation_time, revenue, reward]

    previous_raw_history = state.raw_material_history[0:14]
    state.raw_material_history[0] = new_raw_material_stock
    state.raw_material_history[1:15] = previous_raw_history
    state = ManufactoringState(new_manufacturing_rate, state.current_stock_raw_material + new_raw_material_stock - (new_possible_stock*require_raw_material_per_product), (new_possible_stock + state.current_product_stock - total_products_to_deliver), state.demand_history, state.raw_material_history)

    # profit_list.append(reward)
    print()
    print("Day number:", day_number+1)
    print("Current stock is:", state.current_product_stock)
    print("agent requests for {} this much products".format(new_possible_stock))
    print("demand for next 15 days is:", state.demand_history)
    print("manufucturing cost:", total_costs)
    print("Profit:", reward)
    print()
    
    day_number += 1
    
    return total_products_to_deliver

def get_manufacturing_graphs():
    global df

    df.to_csv("Manufacturing_details.csv")

    fig, axs = plt.subplots(5, figsize=(16, 12))

    axs[0].plot([i for i in range(len(df['current_stock_of_products']))], df['current_stock_of_products'], label='Current Stock')
    axs[0].plot([i for i in range(len(df['new_stock_of_products']))], df['new_stock_of_products'], label='New Stock')
    axs[0].plot([i for i in range(len(df['demand']))], df['demand'], label='Demand')
    axs[0].plot([i for i in range(len(df['total_products_delivered']))], df['total_products_delivered'], label='Total delivered products')
    axs[0].get_xaxis().set_visible(False)
    axs[0].set_title('Current Stock, New Stock and Demand over Time')
    axs[0].legend()

    axs[1].plot([i for i in range(len(np.cumsum(df['production_cost'])))], np.cumsum(df['production_cost']), label='Total production cost')
    axs[1].plot([i for i in range(len(np.cumsum(df['revenue'])))], np.cumsum(df['revenue']), label='Total revenue')
    axs[1].get_xaxis().set_visible(False)
    axs[1].set_title('Production Cost and Revenue Generated over Time')
    axs[1].legend()

    axs[2].plot([i for i in range(len(np.cumsum(df['profit'])))], np.cumsum(df['profit']), label='Total Profit')
    axs[2].get_xaxis().set_visible(False)
    axs[2].set_xlabel('Days')
    axs[2].set_title('Profit Generated over Time')
    axs[2].legend()


    axs[3].plot([i for i in range(len((df['new_stock_of_raw_material'])))], (df['new_stock_of_raw_material']), label='New raw material order')
    axs[3].plot([i for i in range(len((df['current_stock_of_raw_material'])))], (df['current_stock_of_raw_material']), label='Current raw material')
    axs[3].get_xaxis().set_visible(False)
    axs[3].set_title('New Raw material stock and current_raw material stock over Time')
    axs[3].legend()

    axs[4].plot([i for i in range(len((df['manufacturing_rate'])))], (df['manufacturing_rate']), label='Manufacturing rate')
    axs[4].set_xlabel('Days')
    axs[4].set_title('Manufacturing Rate over time')
    axs[4].legend()
    

    return plt
