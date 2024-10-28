package com.devicepricing;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.SequenceGenerator;
import jakarta.persistence.Column;
import jakarta.persistence.Table;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


/**
 * Entity class representing a device
 * Contains all device specifications and the predicted price range
 */
@Entity
@Table(name = "devices")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Device {
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "devices_seq")
    @SequenceGenerator(name = "devices_seq", sequenceName = "devices_id_seq", allocationSize = 1)
    private Long id;

    @NotNull(message = "Battery power is required")
    @Min(value = 1, message = "Battery power must be positive")
    @Column(name = "battery_power")
    private Integer batteryPower;

    @NotNull(message = "Bluetooth field is required")
    @Column(name = "bluetooth")
    private Boolean bluetooth;

    @NotNull(message = "Clock speed is required")
    @Min(value = 0, message = "Clock speed must be non-negative")
    @Column(name = "clock_speed")
    private Double clockSpeed;

    @NotNull(message = "Dual SIM field is required")
    @Column(name = "dual_sim")
    private Boolean dualSim;

    @NotNull(message = "Front camera is required")
    @Min(value = 0, message = "Front camera megapixels must be non-negative")
    @Column(name = "front_camera")
    private Integer frontCamera;

    @NotNull(message = "4G field is required")
    @Column(name = "four_g")
    private Boolean fourG;

    @NotNull(message = "Internal memory is required")
    @Min(value = 0, message = "Internal memory must be non-negative")
    @Column(name = "internal_memory")
    private Integer internalMemory;

    @NotNull(message = "Mobile depth is required")
    @Min(value = 0, message = "Mobile depth must be non-negative")
    @Column(name = "mobile_depth")
    private Double mobileDepth;

    @NotNull(message = "Mobile weight is required")
    @Min(value = 1, message = "Mobile weight must be positive")
    @Column(name = "mobile_weight")
    private Integer mobileWeight;

    @NotNull(message = "Number of cores is required")
    @Min(value = 1, message = "Number of cores must be positive")
    @Column(name = "num_cores")
    private Integer numCores;

    @NotNull(message = "Primary camera is required")
    @Min(value = 0, message = "Primary camera megapixels must be non-negative")
    @Column(name = "primary_camera")
    private Integer primaryCamera;

    @NotNull(message = "Pixel height is required")
    @Min(value = 0, message = "Pixel height must be non-negative")
    @Column(name = "pixel_height")
    private Integer pixelHeight;

    @NotNull(message = "Pixel width is required")
    @Min(value = 0, message = "Pixel width must be non-negative")
    @Column(name = "pixel_width")
    private Integer pixelWidth;

    @NotNull(message = "RAM is required")
    @Min(value = 1, message = "RAM must be positive")
    @Column(name = "ram")
    private Integer ram;

    @NotNull(message = "Screen height is required")
    @Min(value = 0, message = "Screen height must be non-negative")
    @Column(name = "screen_height")
    private Integer screenHeight;

    @NotNull(message = "Screen width is required")
    @Min(value = 0, message = "Screen width must be non-negative")
    @Column(name = "screen_width")
    private Integer screenWidth;

    @NotNull(message = "Talk time is required")
    @Min(value = 1, message = "Talk time must be positive")
    @Column(name = "talk_time")
    private Integer talkTime;

    @NotNull(message = "3G field is required")
    @Column(name = "three_g")
    private Boolean threeG;

    @NotNull(message = "Touch screen field is required")
    @Column(name = "touch_screen")
    private Boolean touchScreen;

    @NotNull(message = "WiFi field is required")
    @Column(name = "wifi")
    private Boolean wifi;

    @Column(name = "price_range")
    private Integer priceRange;


}