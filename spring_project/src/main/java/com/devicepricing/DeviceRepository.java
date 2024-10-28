package com.devicepricing;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

/**
 * Repository interface for Device entity
 * Provides database operations for Device objects
 */
@Repository
public interface DeviceRepository extends JpaRepository<Device, Long> {

    boolean existsById(@SuppressWarnings("null") Long id);

    @Query("SELECT d FROM Device d WHERE d.priceRange = :priceRange")
    List<Device> findByPriceRange(@Param("priceRange") Integer priceRange);

    @Query("SELECT d FROM Device d WHERE d.ram >= :minRam")
    List<Device> findByMinimumRam(@Param("minRam") Integer minRam);
    
}
