#include <Arduino.h>
#include <WiFi.h>
#include <WifiUDP.h>
#include <WiFiServer.h>

#include "motorgo_mini.h"

MotorGo::MotorGoMini motorgo_mini;
MotorGo::MotorChannel& motor_right = motorgo_mini.ch1;
MotorGo::MotorParameters motor_params_right;
MotorGo::PIDParameters velocity_pid_params_right;

// wifiStuff
const char* ssid = "NotARobot";
const char* password = "M1crowave!";
const char* testReplyPacket = "MotorGoHello";

// WiFiUDP udp;
// unsigned int localUdpPort = 4210;
unsigned int port = 4210;
WiFiServer server(4210);
WiFiClient client;

char incomingPacket[255];
char replyPacket[] = "acknowledged";

void setup()
{
  delay(2000);
  Serial.begin(115200);
  delay(3000);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  server.begin();
  Serial.println("wifi connected (udp)");
  Serial.print("IP addy: ");
  Serial.println(WiFi.localIP());
  Serial.printf("UDP server on port %d\n", port);

  motor_params_right.pole_pairs = 11;
  motor_params_right.power_supply_voltage = 5.0;
  motor_params_right.voltage_limit = 5.0;
  motor_params_right.current_limit = 300;
  motor_params_right.velocity_limit = 100.0;
  motor_params_right.calibration_voltage = 2.0;
  motor_params_right.reversed = true;

  // Setup Ch0
  bool calibrate = false;
  motor_right.init(motor_params_right, calibrate);

  // Set velocity controller parameters
  // Setup PID parameters

  velocity_pid_params_right.p = 1.6;
  velocity_pid_params_right.i = 0.01;
  velocity_pid_params_right.d = 0.0;
  velocity_pid_params_right.output_ramp = 10000.0;
  velocity_pid_params_right.lpf_time_constant = 0.11;

  motor_right.set_velocity_controller(velocity_pid_params_right);

  //   Set closed-loop velocity mode
  motor_right.set_control_mode(MotorGo::ControlMode::Voltage);

  //   Enable motors
  motor_right.enable();
}

void loop(){
  // Check if a client has connected
  if (!client.connected()) {
    client = server.available();
  }
  
  if (client) {
    if (client.available()) {
        String message = client.readStringUntil('\n');
    // Serial.printf("Received packet: %s\n", incomingPacket);

    // TODO: Parse the incomingPacket to extract motor power levels and act accordingly
    // cast packet content to double
    float command = (atof(message.c_str()) - 50.0f)/40.0f;
    if (command < (-2.50)) command = -2.50;
    if (command > (2.5)) command = 2.50;

    // Sending a reply (for example, position and velocity data)
    float position = motor_right.get_position();  // Placeholder for actual position data
    float velocity = motor_right.get_velocity();  // Placeholder for actual velocity data
    String reply = String(position) + "," + String(velocity); // Creating a reply string with position and velocity

    client.print(reply);
    // udp.beginPacket(udp.remoteIP(), udp.remotePort());
    // udp.write((const uint8_t*)reply.c_str(), reply.length()); // Corrected: Now actually sending the reply
    // udp.endPacket();

    motor_right.set_target_voltage(command);
  }

  motor_right.loop();
}
}