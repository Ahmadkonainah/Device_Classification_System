package com.devicepricing;

import java.util.List;

import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;

/**
 * REST Controller for Device operations
 * Provides endpoints for device management and price prediction
 */
@RestController
@RequestMapping("/devices")
@Slf4j
public class DeviceController {
    
    private final DeviceService deviceService;
    
    public DeviceController(DeviceService deviceService) {
        this.deviceService = deviceService;
    }
    
    @GetMapping
    public ResponseEntity<?> getAllDevices() {
        try {
            List<Device> devices = deviceService.getAllDevices();
            return ResponseEntity.ok(devices);
        } catch (Exception e) {
            log.error("Error getting all devices", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(500, "Error retrieving devices: " + e.getMessage()));
        }
    }
    
    /**
     * Get device by ID
     * @param id Device ID
     * @return Device if found
     */
    @GetMapping("/{id}")
    public ResponseEntity<?> getDevice(@PathVariable Long id) {
        try {
            log.info("Fetching device with id: {}", id);
            Device device = deviceService.getDevice(id);
            return ResponseEntity.ok(device);
        } catch (ResourceNotFoundException e) {
            log.error("Device not found with id: {}", id);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new ErrorResponse(404, e.getMessage()));
        } catch (Exception e) {
            log.error("Error fetching device with id: {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(500, "Error fetching device: " + e.getMessage()));
        }
    }
    
    /**
     * Create new device
     * @param device Device to create
     * @return Created device
     */
    @PostMapping
        public ResponseEntity<?> createDevice(@Valid @RequestBody Device device) {
            log.info("Creating device with id: {}", device.getId());
            if (device.getId() != null) {
                return ResponseEntity.badRequest().body("ID must not be set for new devices.");
            }
            try {
                Device createdDevice = deviceService.createDevice(device);
                return new ResponseEntity<>(createdDevice, HttpStatus.CREATED);
            } catch (DataIntegrityViolationException e) {
                return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(new ErrorResponse(409, "A device with this ID already exists"));
            }
        }

    /**
     * Update existing device
     * @param id Device ID
     * @param device Updated device data
     * @return Updated device
     */
    @PutMapping("/{id}")
    public ResponseEntity<?> updateDevice(
            @PathVariable Long id,
            @Valid @RequestBody Device device) {
        try {
            log.info("Updating device with id: {}", id);
            Device updatedDevice = deviceService.updateDevice(id, device);
            return ResponseEntity.ok(updatedDevice);
        } catch (ResourceNotFoundException e) {
            log.error("Device not found with id: {}", id);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new ErrorResponse(404, e.getMessage()));
        } catch (IllegalArgumentException e) {
            log.error("Validation error updating device with id: {}", id, e);
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ErrorResponse(400, e.getMessage()));
        } catch (Exception e) {
            log.error("Error updating device with id: {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(500, "Error updating device: " + e.getMessage()));
        }
    }
    
    /**
     * Delete device
     * @param id Device ID
     * @return No content on success
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteDevice(@PathVariable Long id) {
        try {
            log.info("Deleting device with id: {}", id);
            deviceService.deleteDevice(id);
            return ResponseEntity.noContent().build();
        } catch (ResourceNotFoundException e) {
            log.error("Device not found with id: {}", id);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new ErrorResponse(404, e.getMessage()));
        } catch (Exception e) {
            log.error("Error deleting device with id: {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(500, "Error deleting device: " + e.getMessage()));
        }
    }
    
    /**
     * Predict price for device
     * @param id Device ID
     * @return Device with predicted price
     */
    @PostMapping("/predict/{id}")
        public ResponseEntity<?> predictPrice(@PathVariable Long id) {
            try {
                log.info("Predicting price for device: {}", id);
                Device device = deviceService.predictPrice(id);
                return ResponseEntity.ok(device);
            } catch (ResourceNotFoundException e) {
                log.error("Device not found with id: {}", id);
                return ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(new ErrorResponse(404, e.getMessage()));
            } catch (DataIntegrityViolationException e) {
                log.error("Data integrity violation while predicting price for device: {}", id, e);
                return ResponseEntity.status(HttpStatus.CONFLICT)
                        .body(new ErrorResponse(409, "Conflict while updating device price"));
            } catch (Exception e) {
                log.error("Error predicting price for device with id: {}", id, e);
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(new ErrorResponse(500, "Error predicting price: " + e.getMessage()));
            }
        }
    
    /**
     * Bulk prediction for devices
     * @param devices List of devices
     * @return List of devices with predicted prices
     */
    @PostMapping("/predict/bulk")
    public ResponseEntity<?> predictBulk(@RequestBody List<@Valid Device> devices) {
        try {
            log.info("Predicting price for multiple devices");
            List<Device> predictedDevices = deviceService.predictBulk(devices);
            return ResponseEntity.ok(predictedDevices);
        } catch (Exception e) {
            log.error("Error in bulk prediction", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(500, "Error predicting prices in bulk: " + e.getMessage()));
        }
    }
}
