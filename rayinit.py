import ray
ray.init()
print(ray.is_initialized())
ray.shutdown()
print(ray.is_initialized())

