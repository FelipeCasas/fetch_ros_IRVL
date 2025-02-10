#!/usr/bin/env python
import socket
import struct
from threading import Thread, Lock
import rospy
from geometry_msgs.msg import WrenchStamped
import time

class Sensor:
	'''Class manager for ATI Force/Torque sensor via UDP/RDT.
	'''
	def __init__(self, ip= "10.42.42.41", f=float(60)):
		'''Initialization

		Args:
			ip (str): The IP address of the Net F/T box.
			f (int): Frequency of ROS publishing (Hz)
		'''
		# Initialization
		self.ip = ip
		self.port = 49152 #UDP/RDT port
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.connect((ip, self.port))
		self.stream = False
		self.cpf= 1000000.0 #Counts per Force (NetFT Configuration)
		self.cpt= 1000000.0 #Counts per Torque (NetFT Configuration)
		self.lock = Lock()
		
		# Initialize ROS node
		rospy.init_node('ft_sensor', anonymous = True)
		self.pub = rospy.Publisher('/gripper/ft_sensor', WrenchStamped, queue_size=10)
		self.rate = rospy.Rate(f)
		self.data = None

		#Reset NetFT software bias
		self.send(0x0042)

	def send(self, command, count = 0):
		'''Send a  command to the NetFT box with specified sample count.

		Args:
			command (int): The RDT command.
			count (int, optional): The sample count to send. Defaults to 0.
		'''
		header = 0x1234
		message = struct.pack('!HHI', header, command, count)
		self.sock.send(message)

	def receive(self): #Receive measurements
		'''Receives and unpacks a response from the Net F/T box.

		This function receives and unpacks an RDT response from the Net F/T
		box and saves it to the data class attribute.

		Returns:
			list of float: The force and torque values received. The first three
				values are the forces recorded, and the last three are the measured
				torques.
		'''
		rawdata = self.sock.recv(1024)
		data = struct.unpack('!IIIiiiiii', rawdata)[3:]
		with self.lock():
			self.data = [data[i] for i in range(6)] # Replace by Software bias
		return self.data

	def receiveHandler(self):
		'''A handler to receive and store data.'''
		while self.stream:
			self.receive()
			

	def startStreaming(self, handler = True):
		'''Start streaming data continuously

		This function commands the Net F/T box to start sending data continuously.
		By default this also starts a new thread with a handler to save all data
		points coming in. These data points can still be accessed with `measurement`,
		`force`, and `torque`. This handler can also be disabled and measurements
		can be received manually using the `receive` function.

		Args:
			handler (bool, optional): If True start the handler which saves data to be
				used with `measurement`, `force`, and `torque`. If False the
				measurements must be received manually. Defaults to True.
		'''
		self.getMeasurements(0) # Signals NetFT to stream
		if handler:
			self.stream = True
			self.receiveThread = Thread(target = self.receiveHandler)
			self.receiveThread.daemon = True
			self.receiveThread.start()
			self.publisherThread = Thread(target = self.publisherHandler)
			self.publisherThread.daemon = True
			self.publisherThread.start()

	def publisherHandler(self):
		while self.stream:
			with self.lock:
				if self.data:
					self.publish_to_ros()
			self.rate.sleep()

	def getMeasurements(self, n):
		'''Request a given number of samples from the sensor

		This function requests a given number of samples from the sensor. These
		measurements must be received manually using the `receive` function.

		Args:
			n (int): The number of samples to request.
		'''
		self.send(2, count = n)

	def stopStreaming(self):
		'''Stop streaming data continuously

		This function stops the sensor from streaming continuously as started using
		`startStreaming`.
		'''
		self.stream = False
		time.sleep(0.1)
		self.send(0) #Sends signal to NetFT to stop
		
	def publish_to_ros(self):
		'''Publish the current force and torque data to ROS topic.
		'''
		#! Missing transformation from step count to force and torque value
		msg = WrenchStamped()
		msg.header.stamp = rospy.Time.now()
		msg.header.frame_id = "ati_link"  
		msg.wrench.force.x = self.data[0]/self.cpf
		msg.wrench.force.y = self.data[1]/self.cpf
		msg.wrench.force.z = self.data[2]/self.cpf
		msg.wrench.torque.x = self.data[3]/self.cpt
		msg.wrench.torque.y = self.data[4]/self.cpt
		msg.wrench.torque.z = self.data[5]/self.cpt
		self.pub.publish(msg)


if __name__ == "__main__":
	frequency = rospy.get_param('/ft_sensor/frequency', 60.0)
	sensor = Sensor(f=frequency)

	try:
		sensor.startStreaming()
		rospy.loginfo("Starting Net FT Streaming at " + str(frequency) +" Hz." )
		rospy.spin() # Keep node alive
	finally:
		sensor.stopStreaming()
		rospy.loginfo("Net FT Streaming stopped")