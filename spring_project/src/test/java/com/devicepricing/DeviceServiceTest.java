package com.devicepricing;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.RestTemplate;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DeviceServiceTest {

    @Mock
    private DeviceRepository deviceRepository;

    @Mock
    private RestTemplate restTemplate;

    private DeviceService deviceService;

    private Device createTestDevice(Long id) {
        Device device = new Device();
        device.setId(id);
        device.setBatteryPower(3500);
        device.setBluetooth(true);
        device.setClockSpeed(2.2);
        device.setDualSim(true);
        device.setFrontCamera(8);
        device.setFourG(true);
        device.setInternalMemory(64);
        device.setMobileDepth(0.8);
        device.setMobileWeight(150);
        device.setNumCores(4);
        device.setPrimaryCamera(12);
        device.setPixelHeight(1800);
        device.setPixelWidth(1080);
        device.setRam(4000);
        device.setScreenHeight(14);
        device.setScreenWidth(7);
        device.setTalkTime(24);
        device.setThreeG(true);
        device.setTouchScreen(true);
        device.setWifi(true);
        device.setPriceRange(null);
        return device;
    }

    @BeforeEach
        public void setUp() {
            MockitoAnnotations.openMocks(this);
            RestTemplateBuilder restTemplateBuilder = new RestTemplateBuilder();
            // Instead of directly injecting the restTemplate, use the restTemplateBuilder
            deviceService = new DeviceService(deviceRepository, restTemplateBuilder, "http://localhost:5000");

            // Use ReflectionTestUtils to replace the restTemplate after the DeviceService object is created
            ReflectionTestUtils.setField(deviceService, "restTemplate", restTemplate);
        }

    @Test
    public void testGetAllDevices() {
        Device device1 = createTestDevice(1L);
        Device device2 = createTestDevice(2L);
        when(deviceRepository.findAll()).thenReturn(Arrays.asList(device1, device2));

        List<Device> devices = deviceService.getAllDevices();

        assertNotNull(devices);
        assertEquals(2, devices.size());
    }

    @Test
    public void testGetDeviceById() {
        Device device = createTestDevice(1L);
        when(deviceRepository.findById(1L)).thenReturn(Optional.of(device));

        Device foundDevice = deviceService.getDevice(1L);

        assertNotNull(foundDevice);
        assertEquals(1L, foundDevice.getId());
    }

    @Test
    public void testGetDeviceById_NotFound() {
        when(deviceRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> deviceService.getDevice(1L));
    }

    @Test
    public void testCreateDevice() {
        Device device = createTestDevice(null);
        Device savedDevice = createTestDevice(1L);
        when(deviceRepository.save(any(Device.class))).thenReturn(savedDevice);

        Device createdDevice = deviceService.createDevice(device);

        assertNotNull(createdDevice);
        assertEquals(3500, createdDevice.getBatteryPower());
    }

    @Test
    public void testUpdateDevice() {
        Device existingDevice = createTestDevice(1L);
        Device updatedDeviceData = createTestDevice(1L);
        updatedDeviceData.setBatteryPower(4000);

        when(deviceRepository.findById(1L)).thenReturn(Optional.of(existingDevice));
        when(deviceRepository.save(any(Device.class))).thenReturn(updatedDeviceData);

        Device updatedDevice = deviceService.updateDevice(1L, updatedDeviceData);

        assertNotNull(updatedDevice);
        assertEquals(4000, updatedDevice.getBatteryPower());
    }

    @Test
    public void testDeleteDevice() {
        Device device = createTestDevice(1L);
        when(deviceRepository.findById(1L)).thenReturn(Optional.of(device));

        assertDoesNotThrow(() -> deviceService.deleteDevice(1L));
    }

    @Test
    public void testDeleteDevice_NotFound() {
        when(deviceRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> deviceService.deleteDevice(1L));
    }

    @Test
    public void testPredictPrice() {
        Device device = createTestDevice(1L);
        when(deviceRepository.findById(1L)).thenReturn(Optional.of(device));

        PredictionResponse predictionResponse = new PredictionResponse(3);
        when(restTemplate.postForEntity(any(String.class), any(), any()))
                .thenReturn(new ResponseEntity<>(predictionResponse, HttpStatus.OK));
        when(deviceRepository.save(any(Device.class))).thenReturn(device);

        Device predictedDevice = deviceService.predictPrice(1L);

        assertNotNull(predictedDevice);
        assertEquals(3, predictedDevice.getPriceRange());
    }

    @Test
    public void testPredictPrice_DeviceNotFound() {
        when(deviceRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> deviceService.predictPrice(1L));
    }
}
