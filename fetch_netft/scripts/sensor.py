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
	def __init__(self, ip= "10.42.42.41"):
		'''
		Args:
			ip (str): The IP address of the Net F/T box.
		'''
		# Initialization
		self.ip = ip
		self.port = 49152 #UDP/RDT port
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.connect((ip, self.port))
		self.stream = False
		self.cpf= 1000000.0 #Counts per Force (NetFT Configuration)
		self.cpt= 1000000.0 #Counts per Torque (NetFT Configuration)
		
		# Initialize ROS node
		rospy.init_node('ft_sensor', anonymous = True)
		self.pub = rospy.Publisher('/gripper/ft_sensor', WrenchStamped, queue_size=10)
		self.data = None

		#Reset NetFT software bias
		self.send(0x0042)

	def send(self, command, count = 0):
		'''Send a  command to the NetFT using UDP/RDT.

		Args:
			command (int): The RDT command.
			count (int, optional): The sample count to send. Defaults to 0.
		'''
		header = 0x1234
		message = struct.pack('!HHI', header, command, count)
		self.sock.send(message)

	def receive(self): #Receive measurements
		'''Receives and unpacks a response from the Net F/T box.'''
		rawdata = self.sock.recv(1024)
		data = struct.unpack('!IIIiiiiii', rawdata)[3:]
		self.data = [data[i] for i in range(6)] # Replace by Software bias
		return 

	def receiveHandler(self):
		'''A handler to receive and store data.'''
		while self.stream:
			self.receive()
			self.publish_to_ros()
			
	def startStreaming(self):
		'''Start Data Stream'''
		self.getMeasurements(0) # Signals NetFT to stream
		rospy.loginfo("NetFt Stream Started" )
		self.stream = True
		self.receiveThread = Thread(target = self.receiveHandler)
		self.receiveThread.daemon = True
		self.receiveThread.start()

	def getMeasurements(self, n):
		'''Request measurements from NetFT

		Args:
			n (int): The number of samples to request. 0 = stream
		'''
		self.send(2, count = n)

	def stopStreaming(self):
		'''Stop NetFT streaming from data continuously
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
	sensor = Sensor()

	try:
		sensor.startStreaming()
		rospy.loginfo("Starting Net FT Streaming.." )
		rospy.spin() # Keep node alive
	finally:
		sensor.stopStreaming()
		rospy.loginfo("Net FT Streaming stopped")