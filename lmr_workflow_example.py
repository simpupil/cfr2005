#!/usr/bin/env python3
"""
CFR LMR Workflow Example

This script demonstrates a complete Last Millennium Reanalysis (LMR) workflow using CFR,
showing step-by-step implementation from proxy data loading to reconstruction validation.

Author: Claude Code Analysis
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cfr

def create_synthetic_data():
    """
    Create synthetic proxy and climate data for demonstration purposes.
    In practice, you would load real data from files.
    """
    print("Creating synthetic test data...")
    
    # Create synthetic temperature field (model prior)
    years = np.arange(1000, 2001)
    lats = np.linspace(-60, 60, 25)
    lons = np.linspace(0, 360, 50)
    
    # Generate realistic temperature patterns with trends and variability
    temp_data = np.zeros((len(years), len(lats), len(lons)))
    
    for i, year in enumerate(years):
        # Add long-term warming trend
        trend = 0.0008 * (year - 1000)  # 0.8°C over 1000 years
        
        # Add decadal variability
        decadal = 0.3 * np.sin(2 * np.pi * (year - 1000) / 30)
        
        # Add spatial patterns
        for j, lat in enumerate(lats):
            for k, lon in enumerate(lons):
                # Basic latitudinal gradient
                base_temp = 15 * np.cos(np.radians(lat))
                
                # Add interannual noise
                noise = np.random.normal(0, 1.5)
                
                temp_data[i, j, k] = base_temp + trend + decadal + noise
    
    # Create xarray DataArray for temperature
    import xarray as xr
    temp_da = xr.DataArray(
        temp_data,
        dims=['time', 'lat', 'lon'],
        coords={'time': years, 'lat': lats, 'lon': lons},
        attrs={'units': 'degrees_C', 'long_name': 'Surface Temperature'}
    )
    
    # Save synthetic prior data
    temp_da.to_netcdf('synthetic_prior_temp.nc')
    
    # Create synthetic instrumental data (shorter period, higher resolution)
    inst_years = np.arange(1880, 2001)
    inst_temp_data = temp_data[-len(inst_years):] + np.random.normal(0, 0.5, 
                                                    (len(inst_years), len(lats), len(lons)))
    
    inst_temp_da = xr.DataArray(
        inst_temp_data,
        dims=['time', 'lat', 'lon'],
        coords={'time': inst_years, 'lat': lats, 'lon': lons},
        attrs={'units': 'degrees_C', 'long_name': 'Instrumental Temperature'}
    )
    inst_temp_da.to_netcdf('synthetic_obs_temp.nc')
    
    # Create synthetic proxy database
    proxy_data = []
    proxy_locations = [
        {'lat': 45.0, 'lon': 120.0, 'ptype': 'tree.trsgi'},
        {'lat': -30.0, 'lon': 150.0, 'ptype': 'coral.d18O'},
        {'lat': 60.0, 'lon': 30.0, 'ptype': 'lake.temperature'},
        {'lat': 0.0, 'lon': 280.0, 'ptype': 'marine.temperature'},
        {'lat': -45.0, 'lon': 320.0, 'ptype': 'ice.d18O'},
    ]
    
    for i, loc in enumerate(proxy_locations):
        # Extract temperature at proxy location
        lat_idx = np.argmin(np.abs(lats - loc['lat']))
        lon_idx = np.argmin(np.abs(lons - loc['lon']))
        
        local_temp = temp_data[:, lat_idx, lon_idx]
        
        # Apply PSM-like transformation based on proxy type
        if 'tree' in loc['ptype']:
            # Tree rings respond to growing season (JJA) temperature
            proxy_values = 0.8 * local_temp + np.random.normal(0, 1.0, len(local_temp))
            # Add some non-linearity for high temperatures
            proxy_values[local_temp > 20] *= 0.9
            
        elif 'coral' in loc['ptype']:
            # Coral d18O inversely related to temperature
            proxy_values = -0.2 * local_temp + np.random.normal(0, 0.3, len(local_temp))
            
        elif 'ice' in loc['ptype']:
            # Ice core d18O temperature relationship
            proxy_values = 0.15 * local_temp + np.random.normal(0, 0.5, len(local_temp))
            
        else:
            # Generic linear relationship
            proxy_values = 0.5 * local_temp + np.random.normal(0, 0.8, len(local_temp))
        
        # Randomly remove some data points to simulate real proxy records
        mask = np.random.random(len(years)) > 0.1  # Keep 90% of data
        valid_years = years[mask]
        valid_values = proxy_values[mask]
        
        proxy_data.append({
            'pid': f'Proxy_{i+1:02d}',
            'ptype': loc['ptype'],
            'lat': loc['lat'],
            'lon': loc['lon'],
            'time': valid_years,
            'value': valid_values,
            'units': 'proxy_units',
        })
    
    return proxy_data

def demonstrate_lmr_workflow():
    """
    Demonstrate complete LMR workflow with synthetic data
    """
    print("="*60)
    print("CFR LMR Workflow Demonstration")
    print("="*60)
    
    # Step 1: Create synthetic data
    proxy_data = create_synthetic_data()
    
    # Step 2: Initialize ReconJob
    print("\nStep 1: Initializing ReconJob...")
    job = cfr.ReconJob(verbose=True)
    
    # Step 3: Create and load proxy database
    print("\nStep 2: Creating ProxyDatabase...")
    records = []
    for pdata in proxy_data:
        record = cfr.ProxyRecord(
            pid=pdata['pid'],
            time=pdata['time'],
            value=pdata['value'],
            lat=pdata['lat'],
            lon=pdata['lon'],
            ptype=pdata['ptype']
        )
        records.append(record)
    
    job.proxydb = cfr.ProxyDatabase(records)
    print(f"Created proxy database with {job.proxydb.nrec} records")
    
    # Step 4: Load climate data
    print("\nStep 3: Loading climate fields...")
    job.load_clim(
        tag='prior',
        path_dict={'tas': 'synthetic_prior_temp.nc'},
        anom_period=[1951, 1980]  # Reference period for anomalies
    )
    
    job.load_clim(
        tag='obs',
        path_dict={'tas': 'synthetic_obs_temp.nc'},
        anom_period=[1951, 1980]
    )
    
    # Step 5: Configure PSM types and seasonality
    print("\nStep 4: Configuring PSM parameters...")
    ptype_psm_dict = {
        'tree.trsgi': 'Linear',
        'coral.d18O': 'Linear',
        'lake.temperature': 'Linear',
        'marine.temperature': 'Linear',
        'ice.d18O': 'Linear',
    }
    
    ptype_season_dict = {
        'tree.trsgi': [6, 7, 8],      # JJA for trees
        'coral.d18O': [1,2,3,4,5,6,7,8,9,10,11,12],  # Annual for corals
        'lake.temperature': [7, 8],    # Peak summer for lakes
        'marine.temperature': [1,2,3,4,5,6,7,8,9,10,11,12], # Annual for marine
        'ice.d18O': [1,2,3,4,5,6,7,8,9,10,11,12],     # Annual for ice cores
    }
    
    # Step 6: Calibrate PSMs
    print("\nStep 5: Calibrating Proxy System Models...")
    job.calib_psms(
        ptype_psm_dict=ptype_psm_dict,
        ptype_season_dict=ptype_season_dict,
        calib_period=[1880, 2000],
        use_predefined_R=False
    )
    
    # Print calibration results
    print("\nPSM Calibration Results:")
    print("-" * 40)
    for pid, pobj in job.proxydb.records.items():
        if hasattr(pobj, 'psm') and hasattr(pobj.psm, 'calib_details'):
            if pobj.psm.calib_details is not None:
                r2 = pobj.psm.calib_details['fitR2adj']
                snr = pobj.psm.calib_details['SNR']
                nobs = pobj.psm.calib_details['nobs']
                seasonality = pobj.psm.calib_details['seasonality']
                print(f"{pid:12s}: R²={r2:.3f}, SNR={snr:.2f}, N={nobs}, Months={seasonality}")
            else:
                print(f"{pid:12s}: Calibration failed (insufficient data)")
    
    # Step 7: Split proxy database for assimilation/evaluation
    print("\nStep 6: Splitting proxy database...")
    job.split_proxydb(
        tag='calibrated',
        assim_frac=0.75,  # 75% for assimilation, 25% for evaluation
        seed=42
    )
    
    assim_count = job.proxydb.nrec_tags(keys=['assim'])
    eval_count = job.proxydb.nrec_tags(keys=['eval'])
    print(f"Assimilation proxies: {assim_count}")
    print(f"Evaluation proxies: {eval_count}")
    
    # Step 8: Run data assimilation
    print("\nStep 7: Running Data Assimilation...")
    job.run_da(
        recon_period=[1200, 1980],    # Reconstruction period
        recon_loc_rad=15000,          # Localization radius (15,000 km)
        recon_timescale=1,            # Annual resolution
        nens=50,                      # Ensemble size (reduced for demo)
        seed=42,
        verbose=True
    )
    
    # Step 9: Analyze results
    print("\nStep 8: Analyzing reconstruction results...")
    temp_recon = job.recon_fields['tas']  # Shape: (time, ensemble, lat, lon)
    print(f"Reconstruction shape: {temp_recon.shape}")
    
    # Calculate ensemble statistics
    recon_mean = np.mean(temp_recon, axis=1)  # Ensemble mean
    recon_std = np.std(temp_recon, axis=1)    # Ensemble spread
    
    # Extract global mean temperature time series
    # Weight by cos(latitude) for proper global averaging
    lats = job.prior['tas'].da.lat.values
    lat_weights = np.cos(np.radians(lats))
    lat_weights_norm = lat_weights / np.sum(lat_weights)
    
    global_mean_recon = np.average(
        np.average(recon_mean, axis=2),  # Average over longitude
        axis=1,                          # Average over latitude
        weights=lat_weights_norm
    )
    
    global_std_recon = np.average(
        np.average(recon_std, axis=2),
        axis=1,
        weights=lat_weights_norm
    )
    
    # Step 10: Validation and visualization
    print("\nStep 9: Validation and Visualization...")
    
    # Create validation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Global mean temperature time series
    recon_years = np.arange(1200, 1981)
    axes[0, 0].plot(recon_years, global_mean_recon, 'b-', label='Reconstruction')
    axes[0, 0].fill_between(recon_years, 
                           global_mean_recon - global_std_recon,
                           global_mean_recon + global_std_recon,
                           alpha=0.3, color='blue', label='±1σ uncertainty')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Global Mean Temperature Anomaly (°C)')
    axes[0, 0].set_title('Reconstructed Global Temperature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Proxy locations and types
    proxy_lats = [pobj.lat for pobj in job.proxydb.records.values()]
    proxy_lons = [pobj.lon for pobj in job.proxydb.records.values()]
    proxy_types = [pobj.ptype for pobj in job.proxydb.records.values()]
    
    # Color code by proxy type
    type_colors = {'tree.trsgi': 'green', 'coral.d18O': 'red', 
                   'lake.temperature': 'blue', 'marine.temperature': 'orange',
                   'ice.d18O': 'cyan'}
    colors = [type_colors.get(pt, 'gray') for pt in proxy_types]
    
    scatter = axes[0, 1].scatter(proxy_lons, proxy_lats, c=colors, s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_title('Proxy Network')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add legend for proxy types
    for ptype, color in type_colors.items():
        axes[0, 1].scatter([], [], c=color, label=ptype, s=50)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: PSM calibration quality
    r2_values = []
    snr_values = []
    proxy_names = []
    
    for pid, pobj in job.proxydb.records.items():
        if (hasattr(pobj, 'psm') and hasattr(pobj.psm, 'calib_details') 
            and pobj.psm.calib_details is not None):
            r2_values.append(pobj.psm.calib_details['fitR2adj'])
            snr_values.append(pobj.psm.calib_details['SNR'])
            proxy_names.append(pid)
    
    if r2_values:
        x_pos = np.arange(len(proxy_names))
        bars = axes[1, 0].bar(x_pos, r2_values, alpha=0.7)
        axes[1, 0].set_xlabel('Proxy Record')
        axes[1, 0].set_ylabel('Adjusted R²')
        axes[1, 0].set_title('PSM Calibration Quality')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(proxy_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Color bars by quality
        for i, (bar, r2) in enumerate(zip(bars, r2_values)):
            if r2 > 0.5:
                bar.set_color('green')
            elif r2 > 0.3:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
    
    # Plot 4: Reconstruction uncertainty over time
    if len(global_std_recon) > 0:
        axes[1, 1].plot(recon_years, global_std_recon, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Reconstruction Uncertainty (°C)')
        axes[1, 1].set_title('Temporal Evolution of Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lmr_results_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 11: Export results summary
    print("\nStep 10: Saving results summary...")
    results_summary = {
        'reconstruction_period': [1200, 1980],
        'ensemble_size': temp_recon.shape[1],
        'spatial_resolution': f"{len(lats)}x{len(job.prior['tas'].da.lon)}",
        'proxy_records_total': job.proxydb.nrec,
        'proxy_records_assim': assim_count,
        'proxy_records_eval': eval_count,
        'global_mean_final': float(global_mean_recon[-1]),
        'global_uncertainty_final': float(global_std_recon[-1]),
        'global_trend_1200_1980': float(global_mean_recon[-1] - global_mean_recon[0]),
    }
    
    # Save summary as JSON
    import json
    with open('lmr_reconstruction_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save reconstruction data
    recon_da = job.prior['tas'].da.copy()
    recon_da.values = recon_mean
    recon_da.to_netcdf('temperature_reconstruction_mean.nc')
    
    uncertainty_da = job.prior['tas'].da.copy()
    uncertainty_da.values = recon_std
    uncertainty_da.to_netcdf('temperature_reconstruction_uncertainty.nc')
    
    print("\n" + "="*60)
    print("LMR RECONSTRUCTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Reconstruction Period: {results_summary['reconstruction_period']}")
    print(f"Final Global Mean Anomaly: {results_summary['global_mean_final']:.3f}°C")
    print(f"Final Global Uncertainty: {results_summary['global_uncertainty_final']:.3f}°C")
    print(f"Total Temperature Change: {results_summary['global_trend_1200_1980']:.3f}°C")
    print("\nOutput Files Generated:")
    print("- lmr_results_demo.png (visualization)")
    print("- lmr_reconstruction_summary.json (metadata)")
    print("- temperature_reconstruction_mean.nc (reconstruction)")
    print("- temperature_reconstruction_uncertainty.nc (uncertainty)")
    print("\nCleanup: Removing temporary synthetic data files...")
    
    # Clean up temporary files
    for temp_file in ['synthetic_prior_temp.nc', 'synthetic_obs_temp.nc']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return job, results_summary

def demonstrate_advanced_features():
    """
    Demonstrate advanced LMR features including Monte Carlo and custom PSMs
    """
    print("\n" + "="*60)
    print("ADVANCED LMR FEATURES DEMONSTRATION")
    print("="*60)
    
    # This would show Monte Carlo reconstruction
    print("\nMonte Carlo Reconstruction:")
    print("- Multiple random proxy splits")
    print("- Ensemble of reconstructions")  
    print("- Robust uncertainty quantification")
    
    example_mc_config = """
    # Monte Carlo configuration example:
    job.run_da_mc(
        recon_period=[1000, 2000],
        recon_seeds=np.arange(0, 100),  # 100 MC iterations
        assim_frac=0.75,                # 75% assimilation, 25% evaluation
        nens=100,                       # 100-member ensemble per iteration
        save_dirpath='./mc_results',
        output_full_ens=False,          # Save ensemble statistics only
        compress_params={'zlib': True}  # Compress NetCDF output
    )
    """
    print(example_mc_config)
    
    print("\nCustom PSM Implementation:")
    custom_psm_example = """
    class SeasonalNonlinearPSM(cfr.psm.Linear):
        '''Custom PSM with seasonal and nonlinear effects'''
        
        def calibrate(self, **kwargs):
            # Multi-seasonal calibration
            seasonal_models = {}
            for season in ['winter', 'spring', 'summer', 'fall']:
                months = get_season_months(season)
                seasonal_models[season] = self.fit_seasonal_model(months)
            
            # Select best seasonal model
            self.best_season = max(seasonal_models, key=lambda x: seasonal_models[x]['r2'])
            self.model = seasonal_models[self.best_season]['model']
            
        def forward(self, **kwargs):
            # Apply nonlinear transformation for extreme values
            linear_prediction = super().forward(**kwargs)
            nonlinear_prediction = self.apply_threshold_effects(linear_prediction)
            return nonlinear_prediction
    """
    print(custom_psm_example)

if __name__ == "__main__":
    """
    Main execution block - run complete LMR demonstration
    """
    try:
        # Set up environment
        print("Setting up CFR LMR demonstration environment...")
        
        # Run main demonstration  
        job, summary = demonstrate_lmr_workflow()
        
        # Show advanced features
        demonstrate_advanced_features()
        
        print("\n🎉 CFR LMR demonstration completed successfully!")
        print("Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
    finally:
        print("\nDemonstration finished.")