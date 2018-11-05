import time

class logger:
	self.fname = "log_{}.txt".format(time.time())
	self.handle = open(self.fname,'wb')

	def log(self,*args,*kwargs):
		print(*args,*kwargs)
		self.handle.write(*args,*kwargs)
		self.handle.write("\n")

	def close(self):
		self.handle.close()
		print("Log saved to {}".format(self.fname))
