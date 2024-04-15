import numpy as np
from typing import List

class Transportation:
  def __init__(self, total_parcels, vehicals_capacity, vehicals_cost, num_of_vehicals_per_type, routes_km, routes_time, alpha):
    self.total_parcels = total_parcels
    self.vehicals_capacity = vehicals_capacity
    self.vehicals_cost = vehicals_cost
    self.num_of_vehicals_per_type = num_of_vehicals_per_type
    self.routes_km = routes_km
    self.routes_time = routes_time
    self.alpha = 0.9

    self.final_data = []
    self.vehical_capacity_per_route = []
    self.time_per_route = []
    self.vehical_cost_per_route  = []

    for i in  range(len(self.routes_km)):
      for j in range(len(self.vehicals_cost)):
          data = {'total_vehical_cost': 0,
            'total_time':0,
            'vehical_number':0,
            'route_number':0,
            'vehicals_capacity':0,
            }
          data['total_vehical_cost'] = self.vehicals_cost[j]*self.routes_km[i]
          data['total_time'] = self.routes_time[i]
          data['vehical_number'] = j
          data['route_number'] = i
          data['vehicals_capacity'] = self.vehicals_capacity[j]
          self.vehical_cost_per_route.append(data['total_vehical_cost'])
          self.vehical_capacity_per_route.append(data['vehicals_capacity'])
          self.time_per_route.append(data['total_time'])
          self.final_data.append(data)

      self.avg_cost = np.mean(self.vehical_cost_per_route)
      self.avg_time = np.mean(self.time_per_route)
      self.cost_per_route = 1/(self.alpha*np.array(self.vehical_cost_per_route)/self.avg_cost  + (1-self.alpha)*np.array(self.time_per_route)/self.avg_time)

  def __backtrack(self, dp):
    i = len(dp)-1
    j = len(dp[0])-1
    used_vehicals = {}
    while i>0 and j>0:
        if dp[i-1][j][0]==dp[i][j][0]:
            i = i-1
        else:
            if self.final_data[i-1]['vehical_number'] in used_vehicals:
              used_vehicals[self.final_data[i-1]['vehical_number']]+=1
            else:
              used_vehicals[self.final_data[i-1]['vehical_number']] = 1

            if used_vehicals[self.final_data[i-1]['vehical_number']]> self.vehicals_capacity[self.final_data[i-1]['vehical_number']]:
              return False
            j = j- self.vehical_capacity_per_route[i-1]
    return True,used_vehicals

  def unboundedKnapsack(self) -> int:
    # write your code here
    dp = []
    for i in range(len(self.vehical_capacity_per_route)+1):
        dp.append([[-1,0] for i in range(self.total_parcels+1)])

    for i in range(len(self.vehical_capacity_per_route)+1):
        dp[i][0] = [0,0]

    dp[0] = [[0,0] for i in range(self.total_parcels+1)]

    for i in range(1, len(self.vehical_capacity_per_route)+1):
        for j in range(1, self.total_parcels+1):
            if self.vehical_capacity_per_route[i-1] <= j:
                if (dp[i-1][j][0] >= dp[i][j-self.vehical_capacity_per_route[i-1]][0] +self.cost_per_route[i-1]):
                    dp[i][j] = [dp[i-1][j][0], 0]
                else:
                    dp[i][j] = [dp[i][j-self.vehical_capacity_per_route[i-1]][0] +self.cost_per_route[i-1], dp[i][j-self.vehical_capacity_per_route[i-1]][1] + 1]

                if not self.__backtrack(dp[:i+1][:j+1])[0]:

                     dp[i][j] = [dp[i-1][j][0], 0]
            else:
                dp[i][j] = [dp[i-1][j][0], 0]


    i = len(self.vehical_capacity_per_route)
    j = self.total_parcels
    route_used = []

    print(dp[-1][-1][0])
    while i>0 and j>0:
        if dp[i-1][j][0]==dp[i][j][0]:
            i = i-1
        else:
            route_used.append(i-1)
            j = j- self.vehical_capacity_per_route[i-1]
    total_cap = 0
    for i in route_used:
      total_cap += self.vehical_capacity_per_route[i]

    if total_cap<self.total_parcels:
      _, used_vehicals= self.__backtrack(dp)
      print(used_vehicals)
      not_used_veh = []
      for i in used_vehicals.keys():
        if used_vehicals[i]<= self.vehicals_capacity[self.final_data[i]['vehical_number']]:
          not_used_veh.append(i)
      indexs = [i for i, data in enumerate(self.final_data) if data['vehical_number'] in not_used_veh]
      change_cost_per_route = self.cost_per_route.copy()
      change_cost_per_route[indexs] = float('-inf')
      print(change_cost_per_route)
      new_idx = np.argmax(change_cost_per_route)
      route_used.append(new_idx)
    return route_used

  def inference(self):
    output = self.unboundedKnapsack()
    total_costs = 0
    max_time = 0
    for i in output:
      print("*"*30)
      data = self.final_data[i]
      print('Vehical num: ', data['vehical_number'])
      print("Route to be used: ", data['route_number'])
      print("time require for this route: ",data['total_time'])
      max_time = max(max_time,data['total_time'])
      print("Cost for this transportation: ",data['total_vehical_cost'])
      total_costs+=data['total_vehical_cost']
    print(" ")
    print("total_cost for this transportation: ", total_costs)
    return total_costs,max_time


import requests
API_KEY = 'pk.ab811b9144ef39135b2c5b162d492367'
#This function returns dictionary contaning list of distance and duration
def get_locationiq_routes(api_key = API_KEY, source="", destination=""):
    source_geocode = geocode_address(api_key, source)
    destination_geocode = geocode_address(api_key, destination)

    data = request_route(api_key, source_geocode, destination_geocode)

    routes_distance = []
    routes_time = []

    for route in data['routes']:
        distance = route['distance']/1000
        duration = route['duration']/60
        routes_distance.append(distance)
        routes_time.append(duration)

    return {"distance": routes_distance, "duration": routes_time}

def geocode_address(api_key, address):
    url = f"https://us1.locationiq.com/v1/search.php?key={api_key}&q={address}&format=json"
    response = requests.get(url)
    data = response.json()
    if data:
        return data[0]
    else:
        raise Exception("Geocoding failed for address:", address)

def request_route(api_key, source_geocode, destination_geocode):
    source_lat = source_geocode['lat']
    source_lon = source_geocode['lon']
    destination_lat = destination_geocode['lat']
    destination_lon = destination_geocode['lon']

    url = f"https://us1.locationiq.com/v1/directions/driving/{source_lon},{source_lat};{destination_lon},{destination_lat}?key={api_key}&alternatives=true"
    print(url)
    response = requests.get(url)
    data = response.json()
    return data