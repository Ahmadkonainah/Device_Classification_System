package com.devicepricing;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import lombok.extern.slf4j.Slf4j;

/**
 * Main Spring Boot Application class
 * Initializes and configures the Spring Boot application
 */
@SpringBootApplication(scanBasePackages = "com.devicepricing")
@ComponentScan(basePackages = "com.devicepricing")
@EnableTransactionManagement
@EnableJpaRepositories
@Slf4j
public class DevicePricingApplication {

    public static void main(String[] args) {
        log.info("Starting Device Pricing Application");
        SpringApplication.run(DevicePricingApplication.class, args);
    }

    /**
     * Initialize demo data for testing
     */
    @Bean
CommandLineRunner init(DeviceService deviceService) {
    return args -> {
        log.info("Initializing demo data");
        try {
            if (deviceService.getAllDevices().isEmpty()) {
                Device device = new Device();
                device.setBatteryPower(1000);
                device.setBluetooth(true);
                device.setClockSpeed(1.5);
                device.setDualSim(false);
                device.setFrontCamera(8);
                device.setFourG(true);
                device.setInternalMemory(64);
                device.setMobileDepth(0.3);
                device.setMobileWeight(150);
                device.setNumCores(4);
                device.setPrimaryCamera(12);
                device.setPixelHeight(1920);
                device.setPixelWidth(1080);
                device.setRam(4096);
                device.setScreenHeight(14);
                device.setScreenWidth(7);
                device.setTalkTime(20);
                device.setThreeG(true);
                device.setTouchScreen(true);
                device.setWifi(true);
                
                deviceService.createDevice(device);
                log.info("Demo data initialized successfully");
            }
        } catch (Exception e) {
            log.error("Error initializing demo data", e);
        }
    };
}
}
