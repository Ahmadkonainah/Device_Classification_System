package com.devicepricing;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import jakarta.transaction.Transactional;

import lombok.extern.slf4j.Slf4j;

/**
 * Service layer for device operations
 * Handles business logic and integration with Python API
 */
@Service
@Transactional
@Slf4j
public class DeviceService {

    private final DeviceRepository deviceRepository;
    private final RestTemplate restTemplate;
    private final String pythonApiUrl;

    public DeviceService(
            DeviceRepository deviceRepository,
            RestTemplateBuilder restTemplateBuilder,
            @Value("${python.api.url}") String pythonApiUrl) {
        this.deviceRepository = deviceRepository;
        this.restTemplate = restTemplateBuilder.build();
        this.pythonApiUrl = pythonApiUrl;
    }

    /**
     * Get all devices
     * @return List of all devices
     */
    public List<Device> getAllDevices() {
        log.info("Fetching all devices");
        return deviceRepository.findAll();
    }

    /**
     * Get device by ID
     * @param deviceId Device ID
     * @return Device if found
     * @throws ResourceNotFoundException if device not found
     */
    public Device getDevice(Long deviceId) {
        log.info("Fetching device with id: {}", deviceId);
        return deviceRepository.findById(deviceId)
                .orElseThrow(() -> new ResourceNotFoundException(
                        "Device not found with id: " + deviceId));
    }

    /**
     * Create new device
     * @param device Device to create
     * @return Created device
     */
    public Device createDevice(Device device) {
        device.setId(null); // Ensure ID is null for new entries
        return deviceRepository.save(device);
    }
    
    
    /**
     * Update existing device
     * @param deviceId Device ID
     * @param device Updated device data
     * @return Updated device
     * @throws ResourceNotFoundException if device not found
     */
    public Device updateDevice(Long deviceId, Device device) {
        log.info("Updating device with id: {}", deviceId);

        Device existingDevice = getDevice(deviceId);

        // Update fields
        existingDevice.setBatteryPower(device.getBatteryPower());
        existingDevice.setBluetooth(device.getBluetooth());
        existingDevice.setClockSpeed(device.getClockSpeed());
        existingDevice.setDualSim(device.getDualSim());
        existingDevice.setFrontCamera(device.getFrontCamera());
        existingDevice.setFourG(device.getFourG());
        existingDevice.setInternalMemory(device.getInternalMemory());
        existingDevice.setMobileDepth(device.getMobileDepth());
        existingDevice.setMobileWeight(device.getMobileWeight());
        existingDevice.setNumCores(device.getNumCores());
        existingDevice.setPrimaryCamera(device.getPrimaryCamera());
        existingDevice.setPixelHeight(device.getPixelHeight());
        existingDevice.setPixelWidth(device.getPixelWidth());
        existingDevice.setRam(device.getRam());
        existingDevice.setScreenHeight(device.getScreenHeight());
        existingDevice.setScreenWidth(device.getScreenWidth());
        existingDevice.setTalkTime(device.getTalkTime());
        existingDevice.setThreeG(device.getThreeG());
        existingDevice.setTouchScreen(device.getTouchScreen());
        existingDevice.setWifi(device.getWifi());

        validateDevice(existingDevice);
        return deviceRepository.save(existingDevice);
    }

    /**
     * Delete device
     * @param deviceId Device ID
     * @throws ResourceNotFoundException if device not found
     */
    public void deleteDevice(Long deviceId) {
        log.info("Deleting device with id: {}", deviceId);
        Device device = getDevice(deviceId);
        deviceRepository.delete(device);
    }

    /**
     * Predict price range for device
     * @param deviceId Device ID
     * @return Device with predicted price
     */
    @SuppressWarnings("null")
    public Device predictPrice(Long deviceId) {
        log.info("Predicting price for device: {}", deviceId);

        Device device = getDevice(deviceId);

        try {
            Map<String, Object> request = new HashMap<>();
            request.put("batteryPower", device.getBatteryPower());
            request.put("bluetooth", device.getBluetooth() ? 1 : 0);
            request.put("clockSpeed", device.getClockSpeed());
            request.put("dualSim", device.getDualSim() ? 1 : 0);
            request.put("frontCamera", device.getFrontCamera());
            request.put("fourG", device.getFourG() ? 1 : 0);
            request.put("internalMemory", device.getInternalMemory());
            request.put("mobileDepth", device.getMobileDepth());
            request.put("mobileWeight", device.getMobileWeight());
            request.put("numCores", device.getNumCores());
            request.put("primaryCamera", device.getPrimaryCamera());
            request.put("pixelHeight", device.getPixelHeight());
            request.put("pixelWidth", device.getPixelWidth());
            request.put("ram", device.getRam());
            request.put("screenHeight", device.getScreenHeight());
            request.put("screenWidth", device.getScreenWidth());
            request.put("talkTime", device.getTalkTime());
            request.put("threeG", device.getThreeG() ? 1 : 0);
            request.put("touchScreen", device.getTouchScreen() ? 1 : 0);
            request.put("wifi", device.getWifi() ? 1 : 0);

            log.debug("Request payload: {}", request);

            ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                    pythonApiUrl + "/predict",
                    request,
                    PredictionResponse.class
            );

            if (response.getBody() != null) {
                device.setPriceRange(response.getBody().getPriceRange());
                return deviceRepository.save(device);
            }

            throw new RuntimeException("Empty response from prediction API");
        } catch (Exception e) {
            log.error("Error predicting price: ", e);
            throw new RuntimeException("Failed to predict price", e);
        }
    }

    /**
     * Bulk prediction for devices
     * @param devices List of devices
     * @return List of devices with predicted prices
     */
    // In DeviceService.java, replace the current predictBulk method with:

    // In DeviceService.java, replace the current predictBulk method with:

    @SuppressWarnings("null")
    public List<Device> predictBulk(List<Device> devices) {
        log.info("Performing bulk prediction for {} devices", devices.size());
        
        try {
            List<Map<String, Object>> requestPayload = devices.stream().map(device -> {
                Map<String, Object> request = new LinkedHashMap<>();
                request.put("batteryPower", device.getBatteryPower());
                request.put("bluetooth", device.getBluetooth() ? 1 : 0);
                request.put("clockSpeed", device.getClockSpeed());
                request.put("dualSim", device.getDualSim() ? 1 : 0);
                request.put("frontCamera", device.getFrontCamera());
                request.put("fourG", device.getFourG() ? 1 : 0);
                request.put("internalMemory", device.getInternalMemory());
                request.put("mobileDepth", device.getMobileDepth());
                request.put("mobileWeight", device.getMobileWeight());
                request.put("numCores", device.getNumCores());
                request.put("primaryCamera", device.getPrimaryCamera());
                request.put("pixelHeight", device.getPixelHeight());
                request.put("pixelWidth", device.getPixelWidth());
                request.put("ram", device.getRam());
                request.put("screenHeight", device.getScreenHeight());
                request.put("screenWidth", device.getScreenWidth());
                request.put("talkTime", device.getTalkTime());
                request.put("threeG", device.getThreeG() ? 1 : 0);
                request.put("touchScreen", device.getTouchScreen() ? 1 : 0);
                request.put("wifi", device.getWifi() ? 1 : 0);
                return request;
            }).collect(Collectors.toList());
    
            @SuppressWarnings("rawtypes")
            ResponseEntity<Map> response = restTemplate.postForEntity(
                pythonApiUrl + "/predict/bulk",
                requestPayload,
                Map.class
            );
    
            if (response.getBody() != null && response.getBody().containsKey("predictions")) {
                @SuppressWarnings("unchecked")
                List<Integer> predictions = (List<Integer>) response.getBody().get("predictions");
                if (predictions.size() == devices.size()) {
                    for (int i = 0; i < devices.size(); i++) {
                        devices.get(i).setPriceRange(predictions.get(i));
                    }
                    return devices;
                }
                throw new RuntimeException("Prediction count mismatch");
            }
            throw new RuntimeException("Invalid response format from prediction API");
        } catch (Exception e) {
            log.error("Error during bulk prediction: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to predict prices in bulk", e);
        }
    }

    /**
     * Validate device data
     * @param device Device to validate
     * @throws IllegalArgumentException if validation fails
     */
    private void validateDevice(Device device) {
        if (device.getBatteryPower() <= 0) {
            throw new IllegalArgumentException("Battery power must be positive");
        }
        if (device.getClockSpeed() <= 0) {
            throw new IllegalArgumentException("Clock speed must be positive");
        }
        if (device.getInternalMemory() < 0) {
            throw new IllegalArgumentException("Internal memory cannot be negative");
        }
        if (device.getRam() <= 0) {
            throw new IllegalArgumentException("RAM must be positive");
        }}}
